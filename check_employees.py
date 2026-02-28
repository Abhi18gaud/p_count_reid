import pickle

with open('employees_database.pkl', 'rb') as f:
    data = pickle.load(f)
    print('Registered employees:')
    for emp_id, emp_data in data.items():
        print(f'  ID {emp_id}: {emp_data.get("name", "Unknown")}')
        print(f'  Face encoding: {"Yes" if emp_data.get("face_encoding") is not None else "No"}')
