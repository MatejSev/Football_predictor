import { BASE_URL } from '../../../constants/index.js';
import { openPageAndNavigate, waitAndClick, waitForSelectorSafe } from '../../index.js';

export const getMatchIdList = async (browser, leagueSeasonUrl, statsType = 'all') => {
  if (statsType === 'all') {c
    const page = await openPageAndNavigate(browser, `${leagueSeasonUrl}/results`);

    while (true) {
      try {
        await waitAndClick(page, 'a[data-testid="wcl-buttonLink"]');
      } catch (error) {
        break;
      }
    }

    await waitForSelectorSafe(page, '.event__match.event__match--static.event__match--twoLine');

      // Původní logika pro 'all' - vrátí všechny zápasy
      const matchIdList = await page.evaluate(() => {
        return Array.from(document.querySelectorAll('.event__match.event__match--static.event__match--twoLine')).map((element) => {
          return element?.id?.replace('g_1_', '');
        });
      });

    await page.close();
    return matchIdList;
  } else if (statsType === 'teamRatings') {
    const page = await openPageAndNavigate(browser, `${leagueSeasonUrl}/fixtures`);

    while (true) {
      try {
        await waitAndClick(page, 'a[data-testid="wcl-buttonLink"]');
      } catch (error) {
        break;
      }
    }

    await waitForSelectorSafe(page, '.event__match.event__match--static.event__match--twoLine');

    // Nová logika pro 'teamRatings' - filtruje zápasy podle kritérií
    const matchIdList = await page.evaluate(() => {
      const today = new Date();
      const tomorrow = new Date(today);
      tomorrow.setDate(today.getDate() + 1);
      
      // Formatování data pro porovnání (DD.MM.)
      const formatDate = (date) => {
        const day = date.getDate().toString().padStart(2, '0');
        const month = (date.getMonth() + 1).toString().padStart(2, '0');
        return `${day}.${month}.`;
      };

      const todayFormatted = formatDate(today);
      const tomorrowFormatted = formatDate(tomorrow);

      const matches = Array.from(document.querySelectorAll('.event__match.event__match--static.event__match--twoLine'));
      
      return matches
        .filter((element) => {
          // Kontrola data - dnes nebo zítra
          const timeElement = element.querySelector('.event__time');
          if (!timeElement) return false;
          
          const timeText = timeElement.textContent.trim();
          const matchDate = timeText.split(' ')[0]; // Získá část "22.08."
          
          const isToday = matchDate === todayFormatted;
          const isTomorrow = matchDate === tomorrowFormatted;
          
          if (!isToday && !isTomorrow) return false;

          return true;
        })
        .map((element) => {
          return element?.id?.replace('g_1_', '');
        });
    });

    await page.close();
    return matchIdList;
  }
};

export const getMatchData = async (browser, matchId, statsType = 'all') => {
  if (statsType === 'all') {
    const page = await openPageAndNavigate(browser, `${BASE_URL}/match/${matchId}/#/match-summary/match-summary`);

    await waitForSelectorSafe(page, '.duelParticipant__startTime');
    await waitForSelectorSafe(page, "div[data-testid='wcl-summaryMatchInformation'] > div");

    const matchData = await extractMatchData(page);
    const information = await extractMatchInformation(page);

    await page.goto(`${BASE_URL}/match/${matchId}/#/match-summary/match-statistics/0`, { waitUntil: 'domcontentloaded' });
    await waitForSelectorSafe(page, "div[data-testid='wcl-statistics']");
    const statistics = await extractMatchStatistics(page);

    await page.goto(`${BASE_URL}/match/${matchId}/#/match-summary/lineups`, { waitUntil: 'domcontentloaded' });
    await waitForSelectorSafe(page, "div[data-testid='wcl-lineups']");
    const lineups = await extractMatchLineups(page); // 🛠 Tuto funkci si doplníš

    await page.close();
    return { ...matchData, information, statistics, lineups };
  } else if (statsType === 'teamRatings') {
    const page = await openPageAndNavigate(browser, `${BASE_URL}/match/${matchId}/#/match-summary/lineups`);
    await waitForSelectorSafe(page, "div[data-testid='wcl-lineups']");

    // 1️⃣ Vytažení hráčů obou týmů
    const players = await page.evaluate(() => {
      const parsePlayers = (formationSelector, team) => {
        const formation = document.querySelector(formationSelector);
        if (!formation) return [];
        
        const playerLinks = Array.from(formation.querySelectorAll('a.wcl-participantName_HhMjB'));
        
        return playerLinks.map(playerLink => {
          const nameElement = playerLink.querySelector('span');
          const name = nameElement ? nameElement.textContent.trim() : '';
          const url = playerLink.href;
          
          return {
            name,
            url,
            team
          };
        }).filter(player => player.name); // Filtrujeme pouze hráče s jménem
      };

      const homePlayers = parsePlayers('.lf__formation:not(.lf__formationAway)', 'home');
      const awayPlayers = parsePlayers('.lf__formationAway', 'away');
      
      return [...homePlayers, ...awayPlayers];
    });
        
    // 2️⃣ Projít hráče a scrapnout jejich ratingy
    async function getPlayerRatings(player) {
      const profilePage = await browser.newPage();
      await profilePage.goto(player.url, { waitUntil: 'domcontentloaded' });

      // Srolovat dolů na "Career" část
      await profilePage.evaluate(() => window.scrollBy(0, document.body.scrollHeight));
      
      // Počkat na načtení career sekce - upravený selektor
      await waitForSelectorSafe(profilePage, ".careerTab__row");

      const ratings = await profilePage.evaluate(() => {
        const rows = Array.from(document.querySelectorAll('.careerTab__row'));
        return rows.map(row => {
          // Získat sezónu z prvního elementu
          const seasonElement = row.querySelector('.careerTab__season');
          const season = seasonElement ? seasonElement.innerText.trim() : '';
          
          // Získat rating z badge elementu
          const ratingElement = row.querySelector('[data-testid="wcl-badgeRating"] span');
          const ratingText = ratingElement ? ratingElement.innerText.trim() : '';
          const rating = ratingText ? parseFloat(ratingText) : null;
          
          return { season, rating };
        }).filter(item => item.season && item.rating !== null); // Filtrovat pouze platné záznamy
      });

      await profilePage.close();

      // Najít aktuální (2025/2026) a minulou sezónu (2024/2025)
      const currentSeasonData = ratings.find(r => r.season === '2025/2026');
      const lastSeasonData = ratings.find(r => r.season === '2024/2025');
      
      const currentSeason = currentSeasonData?.rating ?? null;
      const lastSeason = lastSeasonData?.rating ?? null;

      // Vypočítat průměr pouze z dostupných ratingů
      const availableRatings = [currentSeason, lastSeason].filter(r => r !== null);
      const avgRating = availableRatings.length > 0 
        ? availableRatings.reduce((a, b) => a + b, 0) / availableRatings.length 
        : null;

      return {
        ...player,
        currentSeason,
        lastSeason,
        avgRating
      };
    }

    console.log(`\n`);
    const playersWithRatings = [];
    for (const player of players) {
      try {
        const p = await getPlayerRatings(player);
        playersWithRatings.push(p);
        console.log(`Zpracován hráč: ${p.name} - Current: ${p.currentSeason}, Last: ${p.lastSeason}, Avg: ${p.avgRating}`);
      } catch (error) {
        console.error(`Chyba při zpracování hráče ${player.name}:`, error);
        // Přidat hráče bez ratingů
        playersWithRatings.push({
          ...player,
          currentSeason: null,
          lastSeason: null,
          avgRating: null
        });
      }
    }

    // 3️⃣ Výpočet průměrů pro tým
    function average(arr) {
      const nums = arr.filter(n => n !== null && !isNaN(n));
      return nums.length ? (nums.reduce((a, b) => a + b, 0) / nums.length).toFixed(2) : null;
    }

    const homePlayersRatings = playersWithRatings
      .filter(p => p.team === 'home')
      .map(p => p.avgRating);

    const awayPlayersRatings = playersWithRatings
      .filter(p => p.team === 'away')
      .map(p => p.avgRating);

    const homeAvg = average(homePlayersRatings);
    const awayAvg = average(awayPlayersRatings);

    console.log('Domácí tým - průměrný rating:', homeAvg);
    console.log('Hostující tým - průměrný rating:', awayAvg);

    await page.close();

    return {
      homeTeamAverageRating: parseFloat(homeAvg),
      awayTeamAverageRating: parseFloat(awayAvg),
      players: playersWithRatings // volitelné, můžeš si nechat pro detail
    };

  }
};

const extractMatchData = async (page) => {
  return await page.evaluate(async () => {
    return {
      stage: document.querySelector('.tournamentHeader__country > a')?.innerText.trim(),
      date: document.querySelector('.duelParticipant__startTime')?.innerText.trim(),
      status: document.querySelector('.fixedHeaderDuel__detailStatus')?.innerText.trim(),
      home: {
        name: document.querySelector('.duelParticipant__home .participant__participantName.participant__overflow')?.innerText.trim(),
        image: document.querySelector('.duelParticipant__home .participant__image')?.src,
      },
      away: {
        name: document.querySelector('.duelParticipant__away .participant__participantName.participant__overflow')?.innerText.trim(),
        image: document.querySelector('.duelParticipant__away .participant__image')?.src,
      },
      result: {
        home: Array.from(document.querySelectorAll('.detailScore__wrapper span:not(.detailScore__divider)'))?.[0]?.innerText.trim(),
        away: Array.from(document.querySelectorAll('.detailScore__wrapper span:not(.detailScore__divider)'))?.[1]?.innerText.trim(),
        regulationTime: document
          .querySelector('.detailScore__fullTime')
          ?.innerText.trim()
          .replace(/[\n()]/g, ''),
        penalties: Array.from(document.querySelectorAll('[data-testid="wcl-scores-overline-02"]'))
          .find((element) => element.innerText.trim().toLowerCase() === 'penalties')
          ?.nextElementSibling?.innerText?.trim()
          .replace(/\s+/g, ''),
      },
    };
  });
};

const extractMatchInformation = async (page) => {
  await page.waitForFunction(() => {
    const elements = document.querySelectorAll("div[data-testid='wcl-summaryMatchInformation'] > div");
    return elements.length >= 5;
  }, { timeout: 10000 });
  
  return await page.evaluate(async () => {
    const elements = Array.from(document.querySelectorAll("div[data-testid='wcl-summaryMatchInformation'] > div"));
    return elements.reduce((acc, element, index) => {
      if (index % 2 === 0) {
        acc.push({
          category: element?.textContent
            .trim()
            .replace(/\s+/g, ' ')
            .replace(/(^[:\s]+|[:\s]+$|:)/g, ''),
          value: elements[index + 1]?.innerText
            .trim()
            .replace(/\s+/g, ' ')
            .replace(/(^[:\s]+|[:\s]+$|:)/g, ''),
        });
      }
      return acc;
    }, []);
  });
};

const extractMatchStatistics = async (page) => {
  return await page.evaluate(async () => {
    return Array.from(document.querySelectorAll("div[data-testid='wcl-statistics']")).map((element) => ({
      category: element.querySelector("div[data-testid='wcl-statistics-category']")?.innerText.trim(),
      homeValue: Array.from(element.querySelectorAll("div[data-testid='wcl-statistics-value'] > strong"))?.[0]?.innerText.trim(),
      awayValue: Array.from(element.querySelectorAll("div[data-testid='wcl-statistics-value'] > strong"))?.[1]?.innerText.trim(),
    }));
  });
};

const extractMatchLineups = async (page) => {
  await page.waitForFunction(() => {
    const ratingDivs = document.querySelectorAll('div[data-testid="wcl-badgeRating"]');
    return ratingDivs.length >= 2;
  }, { timeout: 11000 });

  return await page.evaluate(() => {
    const ratingDivs = Array.from(document.querySelectorAll('div[data-testid="wcl-badgeRating"]'));

    const ratings = ratingDivs.map(div => {
      const span = div.querySelector('span[data-testid="wcl-scores-caption-05"]');
      const rating = span?.innerText?.trim();
      const isHome = div.classList.contains('lf__teamRatingWrapper--home');
      const isAway = div.classList.contains('lf__teamRatingWrapper--away');

      return {
        type: isHome ? 'home' : isAway ? 'away' : 'unknown',
        rating: rating ? parseFloat(rating) : null
      };
    });

    const homeTeamAverageRating = ratings.find(r => r.type === 'home')?.rating ?? null;
    const awayTeamAverageRating = ratings.find(r => r.type === 'away')?.rating ?? null;

    return {
      homeTeamAverageRating,
      awayTeamAverageRating
    };
  });
};

