import time
from random import randint
import numpy as np
from sklearn.linear_model import LinearRegression

def get_user_coefficients():
    """
    Prompts the user to enter a series of coefficients.
    Validates that the number of coefficients is between 4 and 8.
    """
    print("--- STEP 1: Define the Equation ---")
    while True:
        try:
            user_input = input("Enter 4 to 8 coefficients separated by spaces (e.g., '2 5 10 3.5'): ")
            
            # Convert string input into a list of floats
            coeffs = [float(x) for x in user_input.split()]
            
            # Check constraints
            if 4 <= len(coeffs) <= 8:
                print(f"Accepted! Your equation has {len(coeffs)} variables.")
                return coeffs
            else:
                print(f"Error: You entered {len(coeffs)} coefficients. Please enter between 4 and 8.")
                
        except ValueError:
            print("Invalid input. Please ensure you enter only numbers separated by spaces.")

def generate_dataset(coefficients, count=1000, limit=100):
    """
    Generates training data that fits the user's equation.
    """
    train_input = []
    train_output = []
    
    num_vars = len(coefficients)
    
    for _ in range(count):
        # Generate random inputs (x1, x2, ...)
        row = [randint(0, limit) for _ in range(num_vars)]
        
        # Calculate 'y' based on the TRUE equation (Dot Product)
        y = sum(c * x for c, x in zip(coefficients, row))
        
        train_input.append(row)
        train_output.append(y)
        
    return train_input, train_output

def main():
    # 1. Setup the equation (User Input with Validation)
    true_coeffs = get_user_coefficients()
    num_vars = len(true_coeffs)
    
    # 2. Generate Training Data
    print(f"\n--- STEP 2: Generating Training Data ---")
    # We generate data that conforms to the user's hidden equation
    TRAIN_INPUT, TRAIN_OUTPUT = generate_dataset(true_coeffs)
    print(f"Successfully generated 1,000 samples based on your coefficients.")

    # 3. User Test Input
    print("\n--- STEP 3: Enter Test Variables ---")
    print(f"Now, enter {num_vars} input values to test the model:")
    
    test_inputs = []
    # Loop to ensure we get exactly the right number of inputs for the test
    while len(test_inputs) != num_vars:
        try:
            val_str = input(f"Enter {num_vars} numbers separated by spaces: ")
            input_vals = [float(x) for x in val_str.split()]
            
            if len(input_vals) == num_vars:
                test_inputs = input_vals
            else:
                print(f"Error: Expected {num_vars} numbers, but got {len(input_vals)}.")
        except ValueError:
            print("Invalid input. Please enter numbers only.")

    # --- START TIMER ---
    # We measure the time taken for the ML workflow (Training + Prediction)
    start_time = time.time()

    # 4. Train the Model
    # We use n_jobs=-1 to utilize all CPU cores for speed
    predictor = LinearRegression(n_jobs=-1)
    predictor.fit(X=TRAIN_INPUT, y=TRAIN_OUTPUT)

    # 5. Predict
    # Reshape input to 2D array as required by sklearn
    # (samples, features) -> (1, num_vars)
    X_TEST = [test_inputs] 
    predicted_outcome = predictor.predict(X=X_TEST)[0]
    
    # --- END TIMER ---
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 6. Calculate Actual (Analytical) Result for comparison
    actual_outcome = sum(c * x for c, x in zip(true_coeffs, test_inputs))

    # 7. Results
    print("\n" + "="*40)
    print("RESULTS")
    print("="*40)
    print(f"Equation Coefficients: {true_coeffs}")
    print(f"Test Inputs:           {test_inputs}")
    print("-" * 20)
    print(f"Actual Value (Math):      {actual_outcome:.4f}")
    print(f"Predicted Value (Model):  {predicted_outcome:.4f}")
    print("-" * 20)
    
    # Formatting time
    if elapsed_time < 0.001:
        print(f"Model Training & Inference Time: {elapsed_time*1_000_000:.2f} microseconds")
    else:
        print(f"Model Training & Inference Time: {elapsed_time:.6f} seconds")

if __name__ == "__main__":
    main()