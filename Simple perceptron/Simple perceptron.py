#number of iterations variable

iteration_number = -1
# Define the specific number (input)
input_number = 4

# Define the desired output (desired result)
desired_output = 32  # The desired output for the given input (4 * weight = 32)

# Initialize weight and learning rate
weight = 1  # Initial weight
learning_rate = 0.001  # Learning rate for adjustment

# Perform progressive adjustment until the output is near the desired output
print("Iteration | Input | Weight | Output")
print("------------------------")
while True:
    iteration_number = iteration_number + 1
    # Calculate the neuron's output using the current weight
    neuron_output = input_number * weight

    # Display the current values
    print(f"{iteration_number:^5} |{input_number:^5} | {weight:^6} | {neuron_output:^6}")

    # Check if the output is within the acceptable range of the desired output
    if 0.999 * desired_output < neuron_output < 1.001 * desired_output:
        print("\nDesired output achieved!")
        break
    if iteration_number>1000:
        print("\nDesired output diverged!")
        break
    # Update weight based on the error
    weight += learning_rate * (desired_output - neuron_output)

    # Display a message indicating the adjustment
    if neuron_output < desired_output:
        print("Adjusting weight to increase output...")
    else:
        print("Adjusting weight to decrease output...")
    
