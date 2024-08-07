from google.cloud import storage


def upload_df(project_id, bucket_name):
    try:
        storage_client = storage.Client()

        bucket = storage_client.bucket(bucket_name)

        new_bucket = storage_client.create_bucket(bucket, location="eu-west")

        print(f"Bucket {new_bucket.name} created successfully.")

    except Exception as e:
        print(f"Error creating bucket: {str(e)}")


if __name__ == "__main__":
    upload_df()
