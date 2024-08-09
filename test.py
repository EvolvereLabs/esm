import boto3
import os

def download_pdb_files(bucket_name, prefix, output_folder):
    s3 = boto3.client('s3')
    paginator = s3.get_paginator('list_objects_v2')
    
    # Create the output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get('Contents', []):
            if obj['Key'].endswith('.pdb'):
                # Extract the filename from the full S3 key
                filename = os.path.basename(obj['Key'])
                local_file_path = os.path.join(output_folder, filename)
                
                print(f"Downloading {obj['Key']} to {local_file_path}...")
                s3.download_file(bucket_name, obj['Key'], local_file_path)

    print(f"All PDB files have been downloaded to {output_folder}")

# Usage
bucket_name = 'protein-binder-design-bucket-01'
prefix = 'python_pipeline/flow_202407302118187769924/'
output_folder = 'binder_designs'

download_pdb_files(bucket_name, prefix, output_folder)