from fastapi import FastAPI
from pydantic import BaseModel
from model_utils import predict_traffic_impact, get_priority_level

app = FastAPI(title="RoadRisk Traffic Impact API")


class TrafficImpactRequest(BaseModel):
    latitude: float
    longitude: float
    temperature_f: float
    humidity: float
    pressure_in: float
    visibility_mi: float
    wind_speed_mph: float
    precipitation_in: float
    hour: int
    day_of_week: int
    month: int
    is_rush_hour: int
    is_weekend: int
    state: str
    weather_condition: str
    wind_direction: str
    sunrise_sunset: str
    amenity: int = 0
    bump: int = 0
    crossing: int = 0
    give_way: int = 0
    junction: int = 0
    no_exit: int = 0
    railway: int = 0
    roundabout: int = 0
    station: int = 0
    stop: int = 0
    traffic_calming: int = 0
    traffic_signal: int = 0


@app.get("/")
def home():
    return {
        "message": "RoadRisk Traffic Impact Severity API is running",
        "endpoint": "/predict-traffic-impact"
    }


@app.post("/predict-traffic-impact")
def predict_traffic_impact_api(request: TrafficImpactRequest):
    result = predict_traffic_impact(
        latitude=request.latitude,
        longitude=request.longitude,
        temperature_f=request.temperature_f,
        humidity=request.humidity,
        pressure_in=request.pressure_in,
        visibility_mi=request.visibility_mi,
        wind_speed_mph=request.wind_speed_mph,
        precipitation_in=request.precipitation_in,
        hour=request.hour,
        day_of_week=request.day_of_week,
        month=request.month,
        is_rush_hour=request.is_rush_hour,
        is_weekend=request.is_weekend,
        state=request.state,
        weather_condition=request.weather_condition,
        wind_direction=request.wind_direction,
        sunrise_sunset=request.sunrise_sunset,
        amenity=request.amenity,
        bump=request.bump,
        crossing=request.crossing,
        give_way=request.give_way,
        junction=request.junction,
        no_exit=request.no_exit,
        railway=request.railway,
        roundabout=request.roundabout,
        station=request.station,
        stop=request.stop,
        traffic_calming=request.traffic_calming,
        traffic_signal=request.traffic_signal
    )

    result["priority_level"] = get_priority_level(
        result["predicted_severity"]
    )

    return result