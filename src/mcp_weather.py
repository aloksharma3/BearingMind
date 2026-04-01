"""
mcp_weather.py — Weather MCP Server
Industrial AI Predictive Maintenance | BearingMind

An MCP server that retrieves ambient weather conditions (temperature,
humidity, wind) from the Open-Meteo API. No API key required.

What this does:
    Bearing health is affected by environmental conditions that
    vibration sensors alone cannot detect:
    - High ambient temperature → lubricant viscosity drops → accelerated wear
    - High humidity → condensation inside bearing housing during cooldown
      → corrosion on raceways (a common root cause of outer race faults)
    - Rapid temperature swings → thermal cycling of bearing components
      → microscopic fatigue cracks

    The RCA agent calls this MCP to add environmental context.
    Example: SHAP says outer race fault + Weather MCP says 38°C / 85% humidity
    → RCA report adds "high humidity may be causing corrosion on outer race,
      verify seal integrity and housing ventilation."

Open-Meteo API:
    Free, open-source weather API. No registration or key needed.
    Docs: https://open-meteo.com/en/docs
    Endpoints used:
      - /v1/forecast  → current + hourly forecast
      - /v1/archive   → historical weather for a specific date range

    NASA IMS test location: University of Cincinnati, Ohio
    Coordinates: 39.1329° N, 84.5150° W

MCP tool exposed:
    get_weather(lat, lon, date)  → dict with temperature, humidity, conditions
    get_weather_impact()         → pre-formatted assessment for RCA agent

Pipeline position:
    SHAP explainer → RCA Agent → [get_weather_impact] → Weather MCP
                                                          ↓
                                                   temperature, humidity,
                                                   bearing health impact
                                                          ↓
                                                   RCA Agent includes
                                                   in fault report

Usage:
    from src.mcp_weather import WeatherMCP

    weather = WeatherMCP()  # defaults to Cincinnati, OH
    weather.fetch()

    # Get current conditions
    conditions = weather.get_current_conditions()

    # Get bearing health impact assessment (what RCA agent calls)
    impact = weather.get_weather_impact()
    print(impact["assessment_text"])
"""

import os
import json
from datetime import datetime, timedelta
from typing import Optional

# ── HTTP requests ─────────────────────────────────────────────────────────────
try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False
    print("requests not installed. pip install requests")


# ── Constants ─────────────────────────────────────────────────────────────────

# NASA IMS test rig location: University of Cincinnati, Ohio
DEFAULT_LAT = 39.1329
DEFAULT_LON = -84.5150
DEFAULT_LOCATION = "Cincinnati, OH (NASA IMS test rig location)"

# Open-Meteo API endpoints
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"

# Bearing health thresholds
# Based on SKF bearing maintenance guidelines and ISO 15243
TEMP_THRESHOLDS = {
    "low_risk":       30,   # °C — normal operating environment
    "moderate_risk":  35,   # °C — monitor lubricant condition
    "high_risk":      40,   # °C — accelerated lubricant degradation
    "critical_risk":  45,   # °C — immediate concern for grease-lubricated bearings
}

HUMIDITY_THRESHOLDS = {
    "low_risk":       50,   # % — minimal condensation risk
    "moderate_risk":  65,   # % — condensation possible during cooldown
    "high_risk":      80,   # % — significant moisture ingress risk
    "critical_risk":  90,   # % — corrosion risk even with seals
}


# ── Weather impact assessment logic ──────────────────────────────────────────

def _assess_temperature_impact(temp_c: float) -> dict:
    """
    Assess bearing health impact from ambient temperature.

    High ambient temperature affects bearings through:
    1. Lubricant viscosity drop — grease thins, film strength decreases
    2. Thermal expansion — changes internal clearance
    3. Oxidation — grease degrades faster at higher temps
       (rule of thumb: grease life halves for every 15°C above 70°C)
    """
    if temp_c >= TEMP_THRESHOLDS["critical_risk"]:
        return {
            "risk_level": "CRITICAL",
            "impact": (
                f"Ambient temperature ({temp_c:.1f}°C) is critically high. "
                f"Lubricant viscosity is significantly reduced — bearing grease "
                f"life halves for every 15°C above rated temperature. "
                f"Verify grease condition immediately. Consider switching to "
                f"high-temperature synthetic lubricant (e.g., Mobilith SHC 460)."
            ),
        }
    elif temp_c >= TEMP_THRESHOLDS["high_risk"]:
        return {
            "risk_level": "HIGH",
            "impact": (
                f"Ambient temperature ({temp_c:.1f}°C) is elevated. "
                f"Lubricant degradation rate is increased. Bearing operating "
                f"temperature will be higher than normal — monitor temperature "
                f"trend. Shorten re-lubrication interval by 30%."
            ),
        }
    elif temp_c >= TEMP_THRESHOLDS["moderate_risk"]:
        return {
            "risk_level": "MODERATE",
            "impact": (
                f"Ambient temperature ({temp_c:.1f}°C) is moderately elevated. "
                f"No immediate risk, but combined with sustained high load, "
                f"bearing temperature may approach lubricant limits. "
                f"Log for trend analysis."
            ),
        }
    else:
        return {
            "risk_level": "LOW",
            "impact": (
                f"Ambient temperature ({temp_c:.1f}°C) is within normal range. "
                f"No temperature-related bearing health concerns."
            ),
        }


def _assess_humidity_impact(humidity_pct: float) -> dict:
    """
    Assess bearing health impact from ambient humidity.

    High humidity affects bearings through:
    1. Condensation — when machine cools below dew point, water forms
       inside the bearing housing, contaminating grease
    2. Corrosion — moisture on raceway surfaces causes pitting,
       particularly on the outer race (stationary, cooler surface)
    3. Hydrogen embrittlement — water decomposition under contact stress
       releases hydrogen that embrittles bearing steel
    """
    if humidity_pct >= HUMIDITY_THRESHOLDS["critical_risk"]:
        return {
            "risk_level": "CRITICAL",
            "impact": (
                f"Humidity ({humidity_pct:.0f}%) is critically high. "
                f"Significant condensation risk inside bearing housing during "
                f"shutdown or cooldown. Moisture contamination accelerates "
                f"outer race corrosion and hydrogen embrittlement. "
                f"Inspect seals immediately. Consider desiccant breathers "
                f"or nitrogen purge for the bearing housing."
            ),
        }
    elif humidity_pct >= HUMIDITY_THRESHOLDS["high_risk"]:
        return {
            "risk_level": "HIGH",
            "impact": (
                f"Humidity ({humidity_pct:.0f}%) is high. Condensation "
                f"likely during equipment cooldown. Verify bearing seals "
                f"are intact and housing ventilation is adequate. "
                f"Check grease for water contamination (milky appearance)."
            ),
        }
    elif humidity_pct >= HUMIDITY_THRESHOLDS["moderate_risk"]:
        return {
            "risk_level": "MODERATE",
            "impact": (
                f"Humidity ({humidity_pct:.0f}%) is moderately elevated. "
                f"Minor condensation risk during extended shutdown periods. "
                f"Ensure equipment runs periodically to evaporate moisture."
            ),
        }
    else:
        return {
            "risk_level": "LOW",
            "impact": (
                f"Humidity ({humidity_pct:.0f}%) is within normal range. "
                f"No moisture-related bearing health concerns."
            ),
        }


def _combined_risk(temp_risk: str, humidity_risk: str) -> str:
    """Determine combined environmental risk level."""
    levels = ["LOW", "MODERATE", "HIGH", "CRITICAL"]
    t_idx = levels.index(temp_risk)
    h_idx = levels.index(humidity_risk)
    # Combined risk is the higher of the two, bumped up by one level
    # if BOTH are elevated (synergistic effect)
    max_idx = max(t_idx, h_idx)
    if t_idx >= 1 and h_idx >= 1:  # both at least moderate
        max_idx = min(max_idx + 1, 3)  # bump up, cap at CRITICAL
    return levels[max_idx]


# ── Weather MCP Server ────────────────────────────────────────────────────────

class WeatherMCP:
    """
    MCP server for ambient weather data retrieval.

    Calls the Open-Meteo API to get temperature and humidity,
    then assesses the impact on bearing health.

    Args:
        lat       : latitude (default: Cincinnati, OH — NASA IMS location)
        lon       : longitude
        location  : human-readable location name
    """

    # ── MCP tool schema ───────────────────────────────────────────────────
    TOOL_SCHEMA = {
        "name": "get_weather_impact",
        "description": (
            "Get current ambient weather conditions and their impact on "
            "bearing health. Returns temperature, humidity, and a risk "
            "assessment for lubricant degradation and corrosion. Use this "
            "when environmental factors may be contributing to a bearing "
            "fault — especially for outer race corrosion, lubricant "
            "degradation, or unexplained temperature rise."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "date": {
                    "type": "string",
                    "description": (
                        "Date to check weather for (YYYY-MM-DD format). "
                        "Defaults to current date. Use past dates for "
                        "historical analysis."
                    ),
                },
            },
            "required": [],
        },
    }

    def __init__(self,
                 lat: float = DEFAULT_LAT,
                 lon: float = DEFAULT_LON,
                 location: str = DEFAULT_LOCATION):
        self.lat = lat
        self.lon = lon
        self.location = location
        self.weather_data_: dict | None = None
        self.is_fetched_ = False

    # ── Data fetching ─────────────────────────────────────────────────────

    def fetch(self, date: str = None) -> "WeatherMCP":
        """
        Fetch weather data from Open-Meteo API.

        Args:
            date : specific date (YYYY-MM-DD) for historical data.
                   If None, fetches current/forecast data.

        Returns:
            self (for chaining)
        """
        if not REQUESTS_AVAILABLE:
            raise ImportError("requests library required. pip install requests")

        if date:
            self.weather_data_ = self._fetch_historical(date)
        else:
            self.weather_data_ = self._fetch_current()

        self.is_fetched_ = True
        return self

    def _fetch_current(self) -> dict:
        """Fetch current weather + today's hourly forecast."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "current": [
                "temperature_2m",
                "relative_humidity_2m",
                "apparent_temperature",
                "wind_speed_10m",
                "weather_code",
            ],
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
            ],
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "forecast_days": 1,
            "timezone": "auto",
        }

        print(f"  Fetching weather for {self.location} ...")
        response = requests.get(FORECAST_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        # Parse current conditions
        current = data.get("current", {})
        hourly = data.get("hourly", {})

        # Compute daily min/max from hourly data
        hourly_temps = hourly.get("temperature_2m", [])
        hourly_humidity = hourly.get("relative_humidity_2m", [])

        result = {
            "source": "Open-Meteo API (current)",
            "location": self.location,
            "lat": self.lat,
            "lon": self.lon,
            "timestamp": current.get("time", datetime.now().isoformat()),
            "temperature_c": current.get("temperature_2m"),
            "apparent_temperature_c": current.get("apparent_temperature"),
            "humidity_pct": current.get("relative_humidity_2m"),
            "wind_speed_kmh": current.get("wind_speed_10m"),
            "weather_code": current.get("weather_code"),
            "daily_temp_min_c": min(hourly_temps) if hourly_temps else None,
            "daily_temp_max_c": max(hourly_temps) if hourly_temps else None,
            "daily_humidity_min_pct": min(hourly_humidity) if hourly_humidity else None,
            "daily_humidity_max_pct": max(hourly_humidity) if hourly_humidity else None,
            "daily_temp_range_c": (
                (max(hourly_temps) - min(hourly_temps))
                if hourly_temps else None
            ),
        }

        print(f"  Temperature: {result['temperature_c']}°C | "
              f"Humidity: {result['humidity_pct']}% | "
              f"Wind: {result['wind_speed_kmh']} km/h")

        return result

    def _fetch_historical(self, date: str) -> dict:
        """Fetch historical weather for a specific date."""
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "start_date": date,
            "end_date": date,
            "hourly": [
                "temperature_2m",
                "relative_humidity_2m",
                "wind_speed_10m",
            ],
            "temperature_unit": "celsius",
            "wind_speed_unit": "kmh",
            "timezone": "auto",
        }

        print(f"  Fetching historical weather for {date} "
              f"at {self.location} ...")
        response = requests.get(ARCHIVE_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()

        hourly = data.get("hourly", {})
        temps = hourly.get("temperature_2m", [])
        humidity = hourly.get("relative_humidity_2m", [])
        wind = hourly.get("wind_speed_10m", [])

        # Use midday (12:00) values as representative, fall back to mean
        mid_idx = 12 if len(temps) > 12 else len(temps) // 2

        result = {
            "source": f"Open-Meteo API (historical: {date})",
            "location": self.location,
            "lat": self.lat,
            "lon": self.lon,
            "timestamp": f"{date}T12:00:00",
            "temperature_c": temps[mid_idx] if temps else None,
            "humidity_pct": humidity[mid_idx] if humidity else None,
            "wind_speed_kmh": wind[mid_idx] if wind else None,
            "daily_temp_min_c": min(temps) if temps else None,
            "daily_temp_max_c": max(temps) if temps else None,
            "daily_humidity_min_pct": min(humidity) if humidity else None,
            "daily_humidity_max_pct": max(humidity) if humidity else None,
            "daily_temp_range_c": (
                (max(temps) - min(temps)) if temps else None
            ),
        }

        print(f"  Temperature: {result['temperature_c']}°C | "
              f"Humidity: {result['humidity_pct']}% | "
              f"Range: {result['daily_temp_min_c']}–"
              f"{result['daily_temp_max_c']}°C")

        return result

    # ── MCP tool implementation ───────────────────────────────────────────

    def get_current_conditions(self) -> dict:
        """
        Return raw weather data.

        If not fetched yet, fetches current weather.
        """
        if not self.is_fetched_:
            self.fetch()
        return self.weather_data_

    def get_weather_impact(self, date: str = None) -> dict:
        """
        Get weather conditions with bearing health impact assessment.

        This is the primary method called by the RCA agent.

        Args:
            date : optional date (YYYY-MM-DD) for historical weather.

        Returns:
            dict with:
                - conditions      : raw weather data
                - temp_impact     : temperature risk assessment
                - humidity_impact : humidity risk assessment
                - combined_risk   : overall environmental risk level
                - assessment_text : pre-formatted text for RCA prompt
                - thermal_cycling : risk from daily temperature swings
        """
        if date or not self.is_fetched_:
            self.fetch(date=date)

        conditions = self.weather_data_
        temp = conditions.get("temperature_c")
        humidity = conditions.get("humidity_pct")

        if temp is None or humidity is None:
            return {
                "conditions": conditions,
                "error": "Weather data incomplete — could not assess impact.",
            }

        # Assess individual impacts
        temp_impact = _assess_temperature_impact(temp)
        humidity_impact = _assess_humidity_impact(humidity)

        # Combined risk
        combined = _combined_risk(
            temp_impact["risk_level"],
            humidity_impact["risk_level"]
        )

        # Thermal cycling assessment
        temp_range = conditions.get("daily_temp_range_c")
        thermal_cycling = self._assess_thermal_cycling(temp_range)

        # Build assessment text for RCA agent prompt
        assessment_text = self._build_assessment_text(
            conditions, temp_impact, humidity_impact,
            combined, thermal_cycling
        )

        return {
            "conditions":       conditions,
            "temp_impact":      temp_impact,
            "humidity_impact":  humidity_impact,
            "combined_risk":    combined,
            "thermal_cycling":  thermal_cycling,
            "assessment_text":  assessment_text,
        }

    # ── Assessment helpers ────────────────────────────────────────────────

    def _assess_thermal_cycling(self, temp_range_c: float | None) -> dict:
        """
        Assess bearing fatigue risk from daily temperature swings.

        Large temperature swings cause:
        - Differential expansion between inner/outer race
        - Cyclic stress on the interference fit
        - Condensation during cooling phase
        """
        if temp_range_c is None:
            return {"risk_level": "UNKNOWN", "impact": "No range data."}

        if temp_range_c >= 20:
            return {
                "risk_level": "HIGH",
                "impact": (
                    f"Daily temperature range ({temp_range_c:.1f}°C) is large. "
                    f"Thermal cycling causes differential expansion between "
                    f"races and increases condensation risk during cooldown. "
                    f"Check bearing clearance and consider continuous operation "
                    f"to minimize thermal cycling."
                ),
            }
        elif temp_range_c >= 12:
            return {
                "risk_level": "MODERATE",
                "impact": (
                    f"Daily temperature range ({temp_range_c:.1f}°C) is moderate. "
                    f"Some thermal cycling stress. Monitor bearing temperature "
                    f"during startup after long shutdowns."
                ),
            }
        else:
            return {
                "risk_level": "LOW",
                "impact": (
                    f"Daily temperature range ({temp_range_c:.1f}°C) is small. "
                    f"Minimal thermal cycling concern."
                ),
            }

    def _build_assessment_text(self, conditions: dict,
                                temp_impact: dict,
                                humidity_impact: dict,
                                combined_risk: str,
                                thermal_cycling: dict) -> str:
        """Build formatted text for the RCA agent prompt."""
        lines = [
            "=== Environmental Conditions (Weather MCP) ===",
            "",
            f"Location: {conditions.get('location', 'Unknown')}",
            f"Timestamp: {conditions.get('timestamp', 'Unknown')}",
            f"Source: {conditions.get('source', 'Unknown')}",
            "",
            f"Temperature: {conditions.get('temperature_c', '?')}°C"
            f" (range: {conditions.get('daily_temp_min_c', '?')}–"
            f"{conditions.get('daily_temp_max_c', '?')}°C)",
            f"Humidity: {conditions.get('humidity_pct', '?')}%"
            f" (range: {conditions.get('daily_humidity_min_pct', '?')}–"
            f"{conditions.get('daily_humidity_max_pct', '?')}%)",
            f"Wind: {conditions.get('wind_speed_kmh', '?')} km/h",
            "",
            f"Overall environmental risk: {combined_risk}",
            "",
            f"Temperature impact ({temp_impact['risk_level']}):",
            f"  {temp_impact['impact']}",
            "",
            f"Humidity impact ({humidity_impact['risk_level']}):",
            f"  {humidity_impact['impact']}",
            "",
            f"Thermal cycling ({thermal_cycling['risk_level']}):",
            f"  {thermal_cycling['impact']}",
        ]

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────────────────

    def stats(self) -> dict:
        """Return server status."""
        return {
            "location": self.location,
            "lat": self.lat,
            "lon": self.lon,
            "is_fetched": self.is_fetched_,
            "last_fetch": (
                self.weather_data_.get("timestamp")
                if self.weather_data_ else None
            ),
        }


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    print("=" * 60)
    print("Weather MCP Server — BearingMind")
    print("=" * 60)

    # Parse optional date argument
    date = sys.argv[1] if len(sys.argv) > 1 else None

    # Initialize and fetch
    weather = WeatherMCP()

    if date:
        print(f"\nFetching historical weather for {date} ...")
    else:
        print("\nFetching current weather ...")

    weather.fetch(date=date)

    # Show raw conditions
    print(f"\n{'─' * 60}")
    print("Raw conditions:")
    conditions = weather.get_current_conditions()
    for k, v in conditions.items():
        print(f"  {k}: {v}")

    # Show bearing health impact
    print(f"\n{'─' * 60}")
    impact = weather.get_weather_impact(date=date)
    print(impact["assessment_text"])

    print(f"\n{'─' * 60}")
    print(f"Combined risk: {impact['combined_risk']}")
    print(f"\nStats: {weather.stats()}")
