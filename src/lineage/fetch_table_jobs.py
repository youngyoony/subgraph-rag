import requests

BASE_URL = "http://localhost:4601/api/v1"


def get_all_jobs_recursive(namespace):
    """Get all jobs including nested table-level jobs"""
    all_jobs = []

    # Get initial job list
    response = requests.get(f"{BASE_URL}/namespaces/{namespace}/jobs")
    if response.status_code != 200:
        return all_jobs

    jobs = response.json().get("jobs", [])
    print(f"\n{namespace}: Found {len(jobs)} top-level jobs")

    for job in jobs:
        job_name = job["name"]
        all_jobs.append((namespace, job_name, job.get("type", "UNKNOWN")))

        # Get job details to check for child jobs
        job_response = requests.get(
            f"{BASE_URL}/namespaces/{namespace}/jobs/{job_name}"
        )
        if job_response.status_code == 200:
            job_details = job_response.json()

            # Check for facets that might indicate child jobs
            facets = job_details.get("facets", {})
            job_type = facets.get("jobType", {}).get("jobType", "")

            print(f"  {job_name}: {job_type}")

    return all_jobs


# Now let's search for all jobs across ALL namespaces
print("=" * 80)
print("SEARCHING FOR ALL JOBS (INCLUDING TABLE-LEVEL)")
print("=" * 80)

# Get ALL namespaces
namespaces_response = requests.get(f"{BASE_URL}/namespaces")
all_namespaces = [ns["name"] for ns in namespaces_response.json().get("namespaces", [])]

print(f"\nFound {len(all_namespaces)} namespaces: {all_namespaces}")

# Search each namespace for jobs with dots in name (table-level jobs)
table_jobs = []

for namespace in all_namespaces:
    try:
        jobs_response = requests.get(f"{BASE_URL}/namespaces/{namespace}/jobs")
        if jobs_response.status_code != 200:
            continue

        jobs = jobs_response.json().get("jobs", [])

        for job in jobs:
            job_name = job["name"]

            # Table-level jobs often have dots in their names
            if "." in job_name:
                table_jobs.append((namespace, job_name))

                # Get details
                job_url = f"{BASE_URL}/namespaces/{namespace}/jobs/{job_name}"
                job_resp = requests.get(job_url)

                if job_resp.status_code == 200:
                    job_data = job_resp.json()

                    inputs = job_data.get("inputs", [])
                    outputs = job_data.get("outputs", [])

                    if inputs or outputs:
                        print(f"\n✓ FOUND LINEAGE: {namespace}/{job_name}")
                        print(f"  Inputs: {len(inputs)}")
                        print(f"  Outputs: {len(outputs)}")

                        # Show first few
                        if inputs:
                            print(f"  Sample input: {inputs[0].get('name', 'unknown')}")
                        if outputs:
                            print(
                                f"  Sample output: {outputs[0].get('name', 'unknown')}"
                            )

    except Exception as e:
        print(f"Error checking {namespace}: {e}")

print(f"\n\nTotal table-level jobs found: {len(table_jobs)}")
