from generate_trip_observations import get_spark_session, persist_date_formatted_trips_with_timestamp

if __name__ == '__main__':
    spark = get_spark_session("NYCTaxiTrips")

    input_path = "data/nyc-taxi-trip-duration/train.csv"
    trips_output_path = "data/trips_csv"

    persist_date_formatted_trips_with_timestamp(input_path, trips_output_path, trip_cnt=1000000, spark=spark)
