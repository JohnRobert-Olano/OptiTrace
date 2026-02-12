
try:
    import mediapipe.python.solutions as solutions
    print("Found mediapipe.python.solutions")
    print(solutions.face_mesh)
except ImportError as e:
    print(f"ImportError: {e}")
except Exception as e:
    print(f"Error: {e}")
