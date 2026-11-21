import joblib
import pandas as pd

MODEL_FILE = "selected_rf_depth26_leaf10.joblib"

model = None


def load_model():
    global model
    if model is None:
        model = joblib.load(MODEL_FILE)
    return model


def get_model_features():
    loaded_model = load_model()
    return list(loaded_model.feature_names_in_)


def predict_traffic_impact(
    latitude,
    longitude,
    temperature_f,
    humidity,
    pressure_in,
    visibility_mi,
    wind_speed_mph,
    precipitation_in,
    hour,
    day_of_week,
    month,
    is_rush_hour,
    is_weekend,
    state,
    weather_condition,
    wind_direction,
    sunrise_sunset,
    amenity=False,
    bump=False,
    crossing=False,
    give_way=False,
    junction=False,
    no_exit=False,
    railway=False,
    roundabout=False,
    station=False,
    stop=False,
    traffic_calming=False,
    traffic_signal=False
):
    loaded_model = load_model()

    input_data = pd.DataFrame([{
        "Start_Lat": latitude,
        "Start_Lng": longitude,
        "Temperature(F)": temperature_f,
        "Humidity(%)": humidity,
        "Pressure(in)": pressure_in,
        "Visibility(mi)": visibility_mi,
        "Wind_Speed(mph)": wind_speed_mph,
        "Precipitation(in)": precipitation_in,
        "Hour": hour,
        "DayOfWeek": day_of_week,
        "Month": month,
        "Is_Rush_Hour": is_rush_hour,
        "Is_Weekend": is_weekend,
        "State": state,
        "Weather_Condition": weather_condition,
        "Wind_Direction": wind_direction,
        "Sunrise_Sunset": sunrise_sunset,
        "Amenity": int(amenity),
        "Bump": int(bump),
        "Crossing": int(crossing),
        "Give_Way": int(give_way),
        "Junction": int(junction),
        "No_Exit": int(no_exit),
        "Railway": int(railway),
        "Roundabout": int(roundabout),
        "Station": int(station),
        "Stop": int(stop),
        "Traffic_Calming": int(traffic_calming),
        "Traffic_Signal": int(traffic_signal)
    }])
    numeric_columns = [
        "Start_Lat", "Start_Lng", "Temperature(F)", "Humidity(%)",
        "Pressure(in)", "Visibility(mi)", "Wind_Speed(mph)",
        "Precipitation(in)", "Hour", "DayOfWeek", "Month",
        "Is_Rush_Hour", "Is_Weekend", "Amenity", "Bump", "Crossing",
        "Give_Way", "Junction", "No_Exit", "Railway", "Roundabout",
        "Station", "Stop", "Traffic_Calming", "Traffic_Signal"
    ]

    for column in numeric_columns:
        input_data[column] = input_data[column].astype(float)

    prediction = loaded_model.predict(input_data)[0]

    if hasattr(loaded_model, "predict_proba"):
        probabilities = loaded_model.predict_proba(input_data)[0]
        confidence = float(max(probabilities))
    else:
        confidence = None

    return {
        "predicted_severity": int(prediction),
        "confidence": confidence,
        "input_data": input_data.iloc[0].to_dict()
    }


def get_priority_level(predicted_severity):
    if predicted_severity <= 1:
        return "Low"
    elif predicted_severity == 2:
        return "Medium"
    elif predicted_severity == 3:
        return "High"
    else:
        return "Critical"


if __name__ == "__main__":
    loaded_model = load_model()
    print("Model loaded successfully.")
    print("Model features:")
    print(get_model_features())

    result = predict_traffic_impact(
        latitude=34.0522,
        longitude=-118.2437,
        temperature_f=70.0,
        humidity=60.0,
        pressure_in=29.92,
        visibility_mi=10.0,
        wind_speed_mph=5.0,
        precipitation_in=0.0,
        hour=18,
        day_of_week=4,
        month=11,
        is_rush_hour=1,
        is_weekend=0,
        state="CA",
        weather_condition="Clear",
        wind_direction="Calm",
        sunrise_sunset="Day",
        junction=True,
        traffic_signal=True
    )

    print("Prediction result:")
    print(result)
    print("Priority:", get_priority_level(result["predicted_severity"]))