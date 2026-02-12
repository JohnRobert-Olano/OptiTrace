
try:
    import mediapipe as mp
    print(f"Mediapipe imported: {mp}")
    print(f"Solutions: {mp.solutions}")
    print(f"Face Mesh: {mp.solutions.face_mesh}")
    print("SUCCESS")
except AttributeError as e:
    print(f"AttributeError: {e}")
except Exception as e:
    print(f"Error: {e}")
