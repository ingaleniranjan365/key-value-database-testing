from synthetic_data_generation.generate_trip_observations import get_spark_session, persist_date_formatted_trips

if __name__ == '__main__':
    spark = get_spark_session("NYCTaxiTrips")

    input_path = "data/nyc-taxi-trip-duration/train.csv"
    trips_output_path = "data/trips_csv"

    persist_date_formatted_trips(input_path, trips_output_path, trip_cnt=1000000, spark=spark)
