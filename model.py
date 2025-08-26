import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
import warnings
import json
warnings.filterwarnings('ignore')

class FootballLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=128, num_layers=2, num_classes=3, dropout=0.2):
        super(FootballLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        lstm_out, _ = self.lstm(x, (h0, c0))
        # Vezmeme pouze poslední výstup
        output = self.dropout(lstm_out[:, -1, :])
        output = self.fc(output)
        return output

def preprocess_data(df):
    """
    Preprocessing dat podle specifikace
    """
    # Kopie dat
    data = df.copy()

    data = data.fillna(0)
    
    # Odstranění nepotřebných sloupců
    columns_to_remove = ['matchId', 'stage', 'status', 'home.image', 'away.image']
    data = data.drop(columns=[col for col in columns_to_remove if col in data.columns])
    
    # Zpracování capacity a attendance - spojení do jednoho příznaku
    if 'capacity' in data.columns and 'attendance' in data.columns:
        # Převod na číselné hodnoty (odstranění mezer z tisíců)
        data['capacity_clean'] = data['capacity'].astype(str).str.replace(' ', '').str.replace(',', '')
        data['attendance_clean'] = data['attendance'].astype(str).str.replace(' ', '').str.replace(',', '')
        
        # Převod na float
        data['capacity_clean'] = pd.to_numeric(data['capacity_clean'], errors='coerce')
        data['attendance_clean'] = pd.to_numeric(data['attendance_clean'], errors='coerce')
        
        # Vytvoření nového příznaku - procento obsazenosti
        data['stadium_occupancy'] = (data['attendance_clean'] / data['capacity_clean']) * 100
        
        # Odstranění původních sloupců
        data = data.drop(['capacity', 'attendance', 'capacity_clean', 'attendance_clean'], axis=1)

    # Zpracování data
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'], format='%d.%m.%Y %H:%M', errors='coerce')
        data['year'] = data['date'].dt.year
        data['month'] = data['date'].dt.month
        data['day_of_week'] = data['date'].dt.dayofweek
        data['hour'] = data['date'].dt.hour
    
    # Zpracování procentuálních hodnot
    percentage_columns = ['ball_possession.home', 'ball_possession.away', 'passes.home', 'passes.away', 'long_passes.home', 'long_passes.away', 'passes_in_final_third.home', 'passes_in_final_third.away', 'crosses.home', 'crosses.away', 'tackles.home', 'tackles.away']
    for col in percentage_columns:
        if col in data.columns:
            data[col] = data[col].astype(str).str.replace('%', '').astype(float) / 100
    
    # Label encoding pro kategorické proměnné
    categorical_columns = ['home.name', 'away.name', 'referee', 'venue']
    label_encoders = {}
    
    for col in categorical_columns:
        if col in data.columns:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            label_encoders[col] = le
    
    # Vytvoření target proměnné (výsledek zápasu)
    if 'result.home' in data.columns and 'result.away' in data.columns:
        data['match_result'] = np.where(data['result.home'] > data['result.away'], 1,  # domácí vyhrají
                                      np.where(data['result.home'] < data['result.away'], 2, 0))  # remíza = 0, hosté = 2
    
    return data, label_encoders

def create_sequences(data, sequence_length=5):
    """
    Vytvoření sekvencí pro LSTM
    """
    # Seřazení podle data
    if 'date' in data.columns:
        data = data.sort_values('date').reset_index(drop=True)
    
    # Příprava feature a target sloupců
    feature_columns = [col for col in data.columns if col not in ['date', 'match_result', 'result.home', 'result.away']]
    
    X_sequences = []
    y_sequences = []
    weights = []

    # Výpočet vah podle stáří dat
    if 'date' in data.columns:
        max_date = data['date'].max()
        data['days_from_latest'] = (max_date - data['date']).dt.days
        # Exponenciální pokles váhy s časem (čím starší, tím menší váha)
        data['weight'] = np.exp(-data['days_from_latest'] / 365)  # pokles na 37% po roce
    else:
        data['weight'] = 1.0
    
    # Vytvoření sekvencí
    for i in range(len(data) - sequence_length + 1):
        sequence_data = data.iloc[i:i+sequence_length]
        
        # Features pro sekvenci
        X_seq = sequence_data[feature_columns].values
        
        # Target je výsledek posledního zápasu v sekvenci
        y_seq = sequence_data['match_result'].iloc[-1]
        
        # Váha je průměr vah v sekvenci
        weight = sequence_data['weight'].mean()
        
        X_sequences.append(X_seq)
        y_sequences.append(y_seq)
        weights.append(weight)
    
    return np.array(X_sequences), np.array(y_sequences), np.array(weights)

def train_model(X_train, y_train, weights_train, X_val, y_val, input_size, 
                num_epochs=100, batch_size=32, learning_rate=0.001):
    """
    Trénování LSTM modelu
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Používá se: {device}")
    
    # Vytvoření modelu
    model = FootballLSTM(input_size=input_size, hidden_size=128, num_layers=2, num_classes=3)
    model = model.to(device)
    
    # Loss funkce a optimizer
    criterion = nn.CrossEntropyLoss(reduction='none')  # pro vlastní váhy
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
    
    # Příprava dat pro DataLoader
    X_train_tensor = torch.FloatTensor(X_train).to(device)
    y_train_tensor = torch.LongTensor(y_train).to(device)
    weights_train_tensor = torch.FloatTensor(weights_train).to(device)
    
    X_val_tensor = torch.FloatTensor(X_val).to(device)
    y_val_tensor = torch.LongTensor(y_val).to(device)
    
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor, weights_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    # Trénování
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_X, batch_y, batch_weights in train_loader:
            optimizer.zero_grad()
            
            outputs = model(batch_X)
            losses = criterion(outputs, batch_y)
            
            # Aplikace vah
            weighted_loss = (losses * batch_weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_train_loss += weighted_loss.item()
        
        # Validace
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_tensor)
            val_loss = criterion(val_outputs, y_val_tensor).mean()
            val_losses.append(val_loss.item())
            
            # Accuracy
            _, predicted = torch.max(val_outputs.data, 1)
            val_accuracy = (predicted == y_val_tensor).float().mean()
        
        train_losses.append(total_train_loss / len(train_loader))
        scheduler.step(val_loss)
        
        if epoch % 10 == 0:
            print(f'Epoch [{epoch}/{num_epochs}], Train Loss: {train_losses[-1]:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}')
    
    return model, train_losses, val_losses

def create_feature_stub(label_encoders, base_df, home_name, away_name, referee, venue, home_rating, away_rating):
    """
    Vytvoří zápas se známými atributy, ostatní statistiky nahradí průměrem daných týmů (home/away).
    """
    # Vstupní sloupce, které nepatří do feature space
    exclude_columns = ['date', 'match_result', 'result.home', 'result.away']
    feature_columns = [col for col in base_df.columns if col not in exclude_columns]
    
    # Převod týmů na kódované hodnoty
    home_encoded = label_encoders['home.name'].transform([home_name])[0] if home_name in label_encoders['home.name'].classes_ else 0
    away_encoded = label_encoders['away.name'].transform([away_name])[0] if away_name in label_encoders['away.name'].classes_ else 0

    # Filtrování podle týmů
    home_team_df = base_df[base_df['home.name'] == home_encoded]
    away_team_df = base_df[base_df['away.name'] == away_encoded]

    # Výpočet průměrů
    avg_home_stats = home_team_df.mean(numeric_only=True).to_dict()
    avg_away_stats = away_team_df.mean(numeric_only=True).to_dict()
    avg_global = base_df[feature_columns].mean(numeric_only=True).to_dict()

    stub_features = {}

    for col in feature_columns:
        if '.home' in col or col == 'attendance':
            stub_features[col] = avg_home_stats.get(col, avg_global.get(col, 0))
        elif '.away' in col:
            stub_features[col] = avg_away_stats.get(col, avg_global.get(col, 0))
        else:
            stub_features[col] = avg_global.get(col, 0)

    # Nastavení známých hodnot
    stub_features['home_rating'] = home_rating
    stub_features['away_rating'] = away_rating

    # Zakódování kategorií
    categorical_inputs = {
        'home.name': home_name,
        'away.name': away_name,
        'referee': referee,
        'venue': venue
    }

    for col, val in categorical_inputs.items():
        if col in label_encoders and val in label_encoders[col].classes_:
            stub_features[col] = label_encoders[col].transform([val])[0]
        else:
            stub_features[col] = 0  # fallback, pokud třída není známa

    return pd.DataFrame([stub_features])

def predict_from_stub(model, stub_df, scaler, device, sequence_length=5):
    """
    Vytvoření falešné sekvence ze stub zápasu a provedení predikce.
    """
    feature_columns = stub_df.columns.tolist()
    repeated = pd.concat([stub_df] * sequence_length, ignore_index=True)
    repeated = repeated[feature_columns].values

    # Škálování
    repeated_scaled = scaler.transform(repeated)
    repeated_scaled = np.expand_dims(repeated_scaled, axis=0)  # (1, seq_len, features)

    match_tensor = torch.FloatTensor(repeated_scaled).to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(match_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        _, predicted = torch.max(outputs.data, 1)
        return predicted.item(), probabilities.cpu().numpy()[0]

# Hlavní funkce pro použití
def main():
    """
    Hlavní funkce - ukázka použití
    """
    # Cesta k adresáři s daty
    data_dir = r'.\FlashscoreScraping\src\data'

    # Seznam souborů, které začínají 'england_premier_league'
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('england_premier_league') and f.endswith('.csv')]

    # Načtení a spojení všech souborů
    all_dataframes = []
    for file in csv_files:
        df = pd.read_csv(os.path.join(data_dir, file))
        all_dataframes.append(df)

    # Spojení všech do jednoho DataFrame
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"Načteno {len(combined_df)} řádků z {len(csv_files)} souborů.")

    # 1. Preprocessing
    processed_df, label_encoders = preprocess_data(combined_df)

    # 2. Vytvoření sekcí
    X, y, weights = create_sequences(processed_df, sequence_length=5)

    # odstranit
    X = np.nan_to_num(X); y = np.nan_to_num(y); weights = np.nan_to_num(weights)

    # 3. Normalizace (standardizace)
    scaler = StandardScaler()

    # Převedení z 3D do 2D pro škálování
    num_samples, seq_len, num_features = X.shape
    X_reshaped = X.reshape(-1, num_features)
    X_scaled = scaler.fit_transform(X_reshaped)
    X_scaled = X_scaled.reshape(num_samples, seq_len, num_features)

    # 4. Rozdělení na trénovací a validační sady
    X_train, X_val, y_train, y_val, weights_train, weights_val = train_test_split(
        X_scaled, y, weights, test_size=0.2, random_state=42
    )

    # 5. Trénování modelu
    model, train_losses, val_losses = train_model(
        X_train, y_train, weights_train, X_val, y_val,
        input_size=num_features, num_epochs=50
    )

    file_path = os.path.join(data_dir, 'england_premier_league_2025_2026.json')
    # Načíst JSON
    with open(file_path, 'r', encoding='utf-8') as f:
        matches = json.load(f)
    
    # Pro každý zápas
    for match_id, match_data in matches.items():
        home_name = match_data.get("homeName", "").replace("\n", " ").strip()
        away_name = match_data.get("awayName", "").replace("\n", " ").strip()
        referee = match_data.get("referee", "").replace("\n", " ").strip()
        venue = match_data.get("venue", "").replace("\n", " ").strip()
        home_rating = match_data.get("homeTeamAverageRating", None)
        away_rating = match_data.get("awayTeamAverageRating", None)
        print(f"  Home: {home_name}")
        print(f"  Away: {away_name}")
        print(f"  Referee: {referee}")
        print(f"  Venue: {venue}")
        print(f"  Home Rating: {home_rating}")
        print(f"  Away Rating: {away_rating}")
        print("-" * 50)

        # --- Predikce ---
        
        stub = create_feature_stub(
            label_encoders,
            processed_df,
            home_name=home_name,
            away_name=away_name,
            referee=referee,
            venue=venue,
            home_rating=home_rating,
            away_rating=away_rating
        )
    
        predicted_class, probabilities = predict_from_stub(
            model, stub, scaler, device='cuda' if torch.cuda.is_available() else 'cpu'
        )
        print("⚽ Predikce zápasu:")
        print("Výsledek (0=remíza, 1=domácí, 2=hosté):", predicted_class)
        print("Pravděpodobnosti [remíza, domácí, hosté]:", probabilities)
        print("\n")

if __name__ == "__main__":
    main()