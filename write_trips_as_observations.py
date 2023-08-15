from generate_trip_observations import (
    get_spark_session, persist_latest_trip_observations, get_trip_schema,
    format_pickup_and_dropoff_timestamps)

if __name__ == '__main__':
    spark = get_spark_session("NYCTaxiTrips")

    trips_input_path = "data/nyc-taxi-trip-duration/train.csv"
    latest_trip_observations_output_path = "data/trip_observations_csv"

    trips_df = spark.read.csv(trips_input_path, header=True, schema=get_trip_schema()).limit(1000000)
    trips_df = format_pickup_and_dropoff_timestamps(trips_df)
    persist_latest_trip_observations(trips_df, latest_trip_observations_output_path)
