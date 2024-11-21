import csv
import random

# Define some expanded sample data
names = ["John Doe", "Jane Smith", "Alice Johnson", "Bob Brown", "Chris Davis", 
         "Sophia Taylor", "Liam Martinez", "Emma Garcia", "Noah Gonzalez", "Olivia Hernandez"]
medical_histories = [
    "Diabetes, Hypertension", "Asthma, Allergy", "None", "Previous surgery: Appendectomy",
    "Arthritis, High cholesterol", "Cancer (in remission)", "Heart disease", 
    "Thyroid issues", "Migraine history", "Chronic sinusitis"
]
current_conditions = [
    "Healthy", "Cold and cough", "Recovering from surgery", "Mild fever", 
    "Chronic back pain", "Anxiety", "Seasonal allergies", "Shortness of breath", 
    "Skin rash", "Gastrointestinal discomfort"
]
medications = [
    "Paracetamol, Vitamin D", "None", "Ibuprofen, Amoxicillin", 
    "Antihistamines, Insulin", "Painkillers", "Omeprazole, Metformin", 
    "Blood thinners, Beta blockers", "Corticosteroids, Antacids", 
    "Nasal spray, Multivitamins", "Aspirin, Statins"
]

# Add some variability to ages and conditions
def random_condition():
    if random.random() > 0.8:  # 20% chance of having multiple conditions
        return random.choice(current_conditions) + " and " + random.choice(current_conditions)
    return random.choice(current_conditions)

def random_medication():
    if random.random() > 0.7:  # 30% chance of having multiple medications
        return random.choice(medications) + ", " + random.choice(medications)
    return random.choice(medications)

# Generate dummy data
rows = []
for _ in range(200):  # Generate 200 rows of dummy data
    rows.append({
        "Patient Name": f"{random.choice(names)} {random.randint(1000, 9999)}",  # Add unique suffix
        "Age": random.randint(1, 100),  # Ages between 1 and 100
        "Medical History": random.choice(medical_histories),
        "Current Conditions": random_condition(),
        "Current Medications": random_medication()
    })

# Write to a CSV file
csv_file = "expanded_dummy_patient_data.csv"
with open(csv_file, mode="w", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=["Patient Name", "Age", "Medical History", "Current Conditions", "Current Medications"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Expanded dummy data saved to {csv_file}")
