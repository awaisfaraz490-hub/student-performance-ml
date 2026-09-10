"""
generate_dataset.py

Generates a realistic synthetic student performance dataset with meaningful
statistical relationships between features and the Final_Score target.

Run once with:
    python generate_dataset.py

Output:
    data/student_performance.csv
"""

import numpy as np
import pandas as pd
import os

# Reproducibility
np.random.seed(42)

N_ROWS = 600

def generate_dataset(n_rows: int = N_ROWS) -> pd.DataFrame:
    """Generate a synthetic but statistically realistic student dataset."""

    student_ids = [f"STU{str(i).zfill(4)}" for i in range(1, n_rows + 1)]

    # --- Base demographic / lifestyle features -----------------------------
    age = np.random.randint(17, 25, size=n_rows)
    gender = np.random.choice(["Male", "Female"], size=n_rows, p=[0.52, 0.48])
    internet_access = np.random.choice(["Yes", "No"], size=n_rows, p=[0.82, 0.18])
    extracurricular = np.random.choice(["Yes", "No"], size=n_rows, p=[0.45, 0.55])

    # --- Core behavioural features (these DRIVE the target) ----------------
    # Study hours per week: most students study a moderate amount
    study_hours = np.clip(np.random.normal(loc=15, scale=6, size=n_rows), 0, 40)

    # Attendance percentage - correlated loosely with study habits
    attendance_noise = np.random.normal(0, 8, size=n_rows)
    attendance = np.clip(60 + study_hours * 1.1 + attendance_noise, 30, 100)

    # Sleep hours - students who study excessively tend to sleep slightly less
    sleep_hours = np.clip(np.random.normal(7, 1.3, size=n_rows) - (study_hours > 25) * 0.6, 3, 10)

    # Previous academic score (prior semester) - baseline ability proxy
    previous_score = np.clip(np.random.normal(65, 15, size=n_rows), 20, 100)

    # Assignments score - influenced by study hours and previous ability
    assignments_score = np.clip(
        0.4 * previous_score + 0.5 * (study_hours / 40 * 100) +
        np.random.normal(0, 8, size=n_rows),
        0, 100
    )

    # Midterm score - influenced by previous score, attendance and study hours
    midterm_score = np.clip(
        0.45 * previous_score + 0.25 * attendance + 0.25 * (study_hours / 40 * 100) +
        np.random.normal(0, 7, size=n_rows),
        0, 100
    )

    # Internet access gives a small positive boost (access to resources)
    internet_bonus = np.where(internet_access == "Yes", 3, -2)

    # Extracurricular activities: mild positive effect (discipline/time-mgmt)
    # but very heavy involvement combined with low study hours hurts slightly
    extracurricular_bonus = np.where(extracurricular == "Yes", 1.5, 0)

    # Sleep effect: too little or too much sleep is slightly detrimental
    sleep_effect = -np.abs(sleep_hours - 7.5) * 1.5

    # --- Final Score: realistic weighted combination + noise ----------------
    final_score = (
        0.20 * previous_score +
        0.20 * midterm_score +
        0.20 * assignments_score +
        0.20 * attendance +
        0.10 * (study_hours / 40 * 100) +
        internet_bonus +
        extracurricular_bonus +
        sleep_effect +
        np.random.normal(0, 5, size=n_rows)
    )
    final_score = np.clip(final_score, 0, 100)

    df = pd.DataFrame({
        "Student_ID": student_ids,
        "Age": age,
        "Gender": gender,
        "Study_Hours": np.round(study_hours, 1),
        "Attendance": np.round(attendance, 1),
        "Assignments_Score": np.round(assignments_score, 1),
        "Midterm_Score": np.round(midterm_score, 1),
        "Previous_Score": np.round(previous_score, 1),
        "Sleep_Hours": np.round(sleep_hours, 1),
        "Internet_Access": internet_access,
        "Extracurricular_Activities": extracurricular,
        "Final_Score": np.round(final_score, 1),
    })

    # Inject a small, realistic amount of missing data (for preprocessing demo)
    for col in ["Attendance", "Sleep_Hours", "Assignments_Score"]:
        missing_idx = np.random.choice(df.index, size=int(0.02 * n_rows), replace=False)
        df.loc[missing_idx, col] = np.nan

    # Inject a handful of duplicate rows (for preprocessing demo)
    duplicate_rows = df.sample(n=5, random_state=1)
    df = pd.concat([df, duplicate_rows], ignore_index=True)

    return df


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)
    dataset = generate_dataset()
    output_path = os.path.join("data", "student_performance.csv")
    dataset.to_csv(output_path, index=False)
    print(f"Dataset generated successfully: {output_path}")
    print(f"Shape: {dataset.shape}")
