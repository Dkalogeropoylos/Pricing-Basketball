# ==========================================
# MULTI-SPORT CONFIGURATION
# ==========================================

DEFAULT_SPORT = "Basketball"
SPORT = DEFAULT_SPORT

SPORTS = [
    "Basketball",
    "Tennis",
    "Football",
    "UFC",
]

BOOKMAKERS = [
    "Stoiximan",
    "Bet365",
    "Novibet",
    "Betsson",
    "Bwin",
    "Pamestoixima",
    "Superbet",
    "Other"
]

# ==========================================
# BASKETBALL
# ==========================================

BASKETBALL_LEAGUES = [
    "NBA",
    "WNBA",
    "EuroLeague",
    "EuroCup",
    "BCL",
    "GBL",
    "BBL",
    "Pro A",
    "NBB",
    "ACB",
    "BSL",
    "Serie A"
]

BASKETBALL_PLAYER_MARKETS = [
    "Points",
    "Rebounds",
    "Offensive Rebounds",
    "Defensive Rebounds",
    "Assists",
    "3PM",
    "3PA",
    "2PM",
    "2PA",
    "FTM",
    "FTA",
    "Steals",
    "Blocks",
    "Turnovers",
    "PRA",
    "PR",
    "PA",
    "RA"
]

BASKETBALL_TEAM_MARKETS = [
    "Points",
    "Rebounds",
    "Offensive Rebounds",
    "Defensive Rebounds",
    "Assists",
    "3PM",
    "3PA",
    "2PM",
    "2PA",
    "FTM",
    "FTA",
    "Steals",
    "Blocks",
    "Turnovers"
]

BASKETBALL_MATCH_MARKETS = [
    "Moneyline",
    "Handicap / Spread",
    "Total Points",
    "Total Rebounds",
    "Total Offensive Rebounds",
    "Total Defensive Rebounds",
    "Total Assists",
    "Total 3PM",
    "Total 3PA",
    "Total 2PM",
    "Total 2PA",
    "Total FTM",
    "Total FTA",
    "Total Steals",
    "Total Blocks",
    "Total Turnovers"
]

BASKETBALL_OUTRIGHT_MARKETS = [
    "Competition Winner",
    "Series Winner",
    "To Reach Final",
    "Final Matchup",
    "Straight Forecast",
    "Top Scorer - Competition",
    "Top Rebounds - Competition",
    "Top Offensive Rebounds - Competition",
    "Top Defensive Rebounds - Competition",
    "Top Assists - Competition",
    "Top 3PM - Competition",
    "Top 3PA - Competition",
    "Top 2PM - Competition",
    "Top 2PA - Competition",
    "Top FTM - Competition",
    "Top FTA - Competition",
    "Top Steals - Competition",
    "Top Blocks - Competition",
    "Top Turnovers - Competition",
    "Top Scorer - Team",
    "Top Rebounds - Team",
    "Top Offensive Rebounds - Team",
    "Top Defensive Rebounds - Team",
    "Top Assists - Team",
    "Top 3PM - Team",
    "Top 3PA - Team",
    "Top 2PM - Team",
    "Top 2PA - Team",
    "Top FTM - Team",
    "Top FTA - Team",
    "Top Steals - Team",
    "Top Blocks - Team",
    "Top Turnovers - Team"
]

BASKETBALL_PERIODS = [
    "Full Game",
    "1st Half",
    "2nd Half",
    "Q1",
    "Q2",
    "Q3",
    "Q4"
]

BASKETBALL_REASONS = [
    "Injury",
    "Minutes / Rotation",
    "Matchup",
    "Projection Edge"
]

# ==========================================
# TENNIS
# ==========================================

TENNIS_LEAGUES = [
    "ATP",
    "WTA",
    "ATP Challenger",
    "WTA 125",
    "ITF Men",
    "ITF Women",
    "ATP Doubles",
    "WTA Doubles",
    "Davis Cup",
    "Billie Jean King Cup"
]

TENNIS_PLAYER_MARKETS = [
    "Aces",
    "Double Faults",
    "Games Won",
    "Break Points Won",
    "Total Points Won"
]

TENNIS_MATCH_MARKETS = [
    "Match Winner",
    "Set Winner",
    "Total Games",
    "Game Handicap",
    "Total Sets"
]

TENNIS_OUTRIGHT_MARKETS = [
    "Tournament Winner",
    "To Reach Final",
    "To Reach Semi-Final",
    "Final Matchup",
    "Straight Forecast"
]

TENNIS_PERIODS = [
    "Full Match",
    "1st Set",
    "2nd Set",
    "3rd Set",
    "4th Set",
    "5th Set"
]

TENNIS_REASONS = [
    "Injury",
    "Matchup",
    "Surface",
    "Form",
    "Rest / Fatigue",
    "Projection Edge"
]


# ==========================================
# FOOTBALL
# ==========================================

FOOTBALL_LEAGUES = [
    "Premier League",
    "La Liga",
    "Serie A",
    "Bundesliga",
    "Ligue 1",
    "UEFA Champions League",
    "UEFA Europa League",
    "UEFA Conference League",
    "Super League Greece",
    "Eredivisie",
    "Primeira Liga",
    "Championship",
    "MLS",
    "Saudi Pro League",
    "FIFA World Cup",
    "UEFA Euro"
]

FOOTBALL_PLAYER_MARKETS = [
    "Goals",
    "Assists",
    "Shots",
    "Shots on Target",
    "Tackles",
    "Fouls Committed",
    "Fouls Won",
    "Cards",
    "Passes",
    "Goalkeeper Saves",
    "Anytime Goalscorer",
    "Player to be Carded"
]

FOOTBALL_TEAM_MARKETS = [
    "Team Total Goals",
    "Team Corners",
    "Team Cards",
    "Team Shots",
    "Team Shots on Target",
    "Team Fouls",
    "Team Offsides",
    "Team to Score"
]

FOOTBALL_MATCH_MARKETS = [
    "1X2",
    "Double Chance",
    "Draw No Bet",
    "Asian Handicap",
    "Total Goals",
    "Both Teams to Score",
    "Total Corners",
    "Total Cards",
    "Total Shots",
    "Total Shots on Target",
    "Total Fouls",
    "Total Offsides"
]

FOOTBALL_OUTRIGHT_MARKETS = [
    "Competition Winner",
    "Group Winner",
    "To Reach Final",
    "Top 4 Finish",
    "Relegation",
    "Top Goalscorer",
    "Top Assists",
    "Final Matchup",
    "Straight Forecast"
]

FOOTBALL_PERIODS = [
    "Full Match",
    "1st Half",
    "2nd Half"
]

FOOTBALL_REASONS = [
    "Injury",
    "Lineup / Rotation",
    "Matchup",
    "Form",
    "Tactical Matchup",
    "Schedule / Fatigue",
    "Motivation",
    "Weather / Conditions",
    "Projection Edge"
]


# ==========================================
# UFC / MMA
# ==========================================

UFC_LEAGUES = [
    "UFC",
    "UFC Fight Night",
    "Dana White's Contender Series"
]

UFC_PLAYER_MARKETS = [
    "Significant Strikes",
    "Takedowns",
    "Knockdowns"
]

UFC_MATCH_MARKETS = [
    "Fight Winner",
    "Total Rounds",
    "Goes the Distance"
]

UFC_OUTRIGHT_MARKETS = [
    "Event / Tournament Winner"
]

UFC_PERIODS = [
    "Full Fight",
    "Round 1",
    "Round 2",
    "Round 3",
    "Round 4",
    "Round 5"
]

UFC_REASONS = [
    "Injury",
    "Matchup",
    "Style Matchup",
    "Striking",
    "Wrestling / Grappling",
    "Cardio / Pace",
    "Form",
    "Projection Edge"
]

SPORT_CONFIG = {
    "Basketball": {
        "leagues": BASKETBALL_LEAGUES,
        "scopes": ["PLAYER", "TEAM", "MATCH", "OUTRIGHT"],
        "markets": {
            "PLAYER": BASKETBALL_PLAYER_MARKETS,
            "TEAM": BASKETBALL_TEAM_MARKETS,
            "MATCH": BASKETBALL_MATCH_MARKETS,
            "OUTRIGHT": BASKETBALL_OUTRIGHT_MARKETS
        },
        "periods": BASKETBALL_PERIODS,
        "reasons": BASKETBALL_REASONS
    },
    "Tennis": {
        "leagues": TENNIS_LEAGUES,
        "scopes": ["PLAYER", "MATCH", "OUTRIGHT"],
        "markets": {
            "PLAYER": TENNIS_PLAYER_MARKETS,
            "MATCH": TENNIS_MATCH_MARKETS,
            "OUTRIGHT": TENNIS_OUTRIGHT_MARKETS
        },
        "periods": TENNIS_PERIODS,
        "reasons": TENNIS_REASONS
    },
    "Football": {
        "leagues": FOOTBALL_LEAGUES,
        "scopes": ["PLAYER", "TEAM", "MATCH", "OUTRIGHT"],
        "markets": {
            "PLAYER": FOOTBALL_PLAYER_MARKETS,
            "TEAM": FOOTBALL_TEAM_MARKETS,
            "MATCH": FOOTBALL_MATCH_MARKETS,
            "OUTRIGHT": FOOTBALL_OUTRIGHT_MARKETS
        },
        "periods": FOOTBALL_PERIODS,
        "reasons": FOOTBALL_REASONS
    },
    "UFC": {
        "leagues": UFC_LEAGUES,
        "scopes": ["PLAYER", "MATCH", "OUTRIGHT"],
        "markets": {
            "PLAYER": UFC_PLAYER_MARKETS,
            "MATCH": UFC_MATCH_MARKETS,
            "OUTRIGHT": UFC_OUTRIGHT_MARKETS
        },
        "periods": UFC_PERIODS,
        "reasons": UFC_REASONS
    }
}

def get_leagues(sport):
    return list(SPORT_CONFIG[sport]["leagues"])

def get_scope_options(sport):
    return list(SPORT_CONFIG[sport]["scopes"])

def get_default_markets(sport, scope):
    return list(SPORT_CONFIG[sport]["markets"].get(scope, []))

def get_periods(sport):
    return list(SPORT_CONFIG[sport]["periods"])

def get_reasons(sport):
    return list(SPORT_CONFIG[sport]["reasons"])

def get_market_style(sport, scope, market):
    if sport == "Basketball":
        if scope == "MATCH" and market == "Moneyline":
            return "winner"
        if scope == "MATCH" and market == "Handicap / Spread":
            return "handicap"
        return "total"

    if sport == "Tennis":
        if scope == "MATCH" and market in ["Match Winner", "Set Winner"]:
            return "winner"
        if scope == "MATCH" and market == "Game Handicap":
            return "handicap"
        return "total"

    if sport == "Football":
        if scope == "MATCH" and market in [
            "1X2",
            "Double Chance",
            "Draw No Bet"
        ]:
            return "winner"

        if scope == "MATCH" and market == "Asian Handicap":
            return "handicap"

        if market in [
            "Both Teams to Score",
            "Team to Score",
            "Anytime Goalscorer",
            "Player to be Carded"
        ]:
            return "yes_no"

        return "total"

    if sport == "UFC":
        if scope == "MATCH" and market == "Fight Winner":
            return "winner"

        if scope == "MATCH" and market == "Goes the Distance":
            return "yes_no"

        return "total"

    return "total"


def get_winner_side_options(sport, market=None):
    if sport == "Tennis":
        return ["Player 1", "Player 2"]

    if sport == "Football":
        if market == "1X2":
            return ["Home", "Draw", "Away"]

        if market == "Double Chance":
            return ["1X", "12", "X2"]

        return ["Home", "Away"]

    if sport == "UFC":
        return ["Fighter 1", "Fighter 2"]

    return ["Home", "Away"]

# Backwards-compatible aliases
LEAGUES = BASKETBALL_LEAGUES
PLAYER_MARKETS = BASKETBALL_PLAYER_MARKETS
TEAM_MARKETS = BASKETBALL_TEAM_MARKETS
MATCH_MARKETS = BASKETBALL_MATCH_MARKETS
OUTRIGHT_MARKETS = BASKETBALL_OUTRIGHT_MARKETS
PERIODS = BASKETBALL_PERIODS
REASONS = BASKETBALL_REASONS
