import os
from pathlib import Path

import pandas as pd
import requests


class ETLTools:
    """Small ETL helpers for API extraction and Pandas transformations."""

    def extract_load(self, url: str, output_folder: str, format: str):
        """Extract JSON from an API and persist it as CSV, JSON, or Parquet."""
        if format not in {"csv", "json", "parquet"}:
            return f"Unsupported format: {format}"

        project_root = Path(__file__).resolve().parent.parent
        requested_output = Path(output_folder)
        output_path = requested_output if requested_output.is_absolute() else project_root / requested_output

        try:
            response = requests.get(url, timeout=(5, 30))
            response.raise_for_status()
            data = response.json()

            # APIs often return a list directly, or wrap records in "results".
            records = data.get("results", data) if isinstance(data, dict) else data
            if isinstance(records, dict):
                records = [records]
            if not isinstance(records, list):
                return "Failed to extract data: API response must be a JSON object or list."

            df = pd.json_normalize(records)
            output_path.mkdir(parents=True, exist_ok=True)
            filename = output_path / f"extracted_data.{format}"

            if format == "csv":
                df.to_csv(filename, index=False)
            elif format == "json":
                df.to_json(filename, orient="records", lines=True)
            else:
                df.to_parquet(filename, index=False)

            return f"Data successfully extracted and saved to {filename}"
        except (requests.exceptions.RequestException, ValueError, OSError) as exc:
            return f"Failed to extract data: {exc}"

    def transform_load_context(self, file_path: str):
        """Return a small preview of a supported local dataset."""
        file_extension = os.path.splitext(file_path)[1].lower()
        if file_extension == ".csv":
            df = pd.read_csv(file_path)
        elif file_extension == ".json":
            df = pd.read_json(file_path, lines=True)
        elif file_extension == ".parquet":
            df = pd.read_parquet(file_path)
        else:
            return f"Unsupported file format: {file_extension}"

        return str(df.head(3))

    def execute_code(self, code: str, context: dict | None = None):
        """Execute generated transformation code with an explicit variable namespace.

        WARNING: this is not a security sandbox. Only run trusted code until this
        operation is moved to an isolated, resource-limited execution environment.
        """
        namespace = {"pd": pd, "os": os}
        if context:
            namespace.update(context)

        try:
            exec(code, namespace, namespace)
            return "Code executed successfully."
        except Exception as exc:
            return f"Failed to execute code: {exc}"


if __name__ == "__main__":
    obj = ETLTools()
    sample_path = Path(__file__).resolve().parent.parent / "data" / "extract" / "extracted_data.csv"
    print(obj.transform_load_context(str(sample_path)))
