import boto3

def list_available_models():
    """
    Comprehensively lists available Bedrock foundation models.
    """
    try:
        # Create Bedrock client
        bedrock = boto3.client('bedrock')
        
        # Retrieve foundation models
        models = bedrock.list_foundation_models()
        
        # Print header
        print("=" * 80)
        print(f"{'MODEL DETAILS':^80}")
        print("=" * 80)
        
        # Sort models by provider and model ID for better readability
        sorted_models = sorted(
            models['modelSummaries'], 
            key=lambda x: (x.get('providerName', ''), x.get('modelId', ''))
        )
        
        # Print detailed information for each model
        for model in sorted_models:
            print("\n" + "-" * 80)
            print(f"Model ID: {model.get('modelId', 'N/A')}")
            print(f"Provider: {model.get('providerName', 'N/A')}")
            print(f"Model Name: {model.get('modelName', 'N/A')}")
            print(f"Input Modalities: {model.get('inputModalities', 'N/A')}")
            print(f"Output Modalities: {model.get('outputModalities', 'N/A')}")
            print(f"Lifecycle Status: {model.get('modelLifecycle', 'N/A')}")
            
            # Add any additional relevant details
            if 'customizationType' in model:
                print(f"Customization Type: {model['customizationType']}")
    
    except Exception as e:
        print(f"Error listing models: {e}")
        print("Possible issues:")
        print("1. AWS CLI not configured")
        print("2. Insufficient IAM permissions")
        print("3. Incorrect AWS region")

def list_model_capabilities():
    """
    Provides a more focused view of model capabilities
    """
    try:
        bedrock = boto3.client('bedrock')
        models = bedrock.list_foundation_models()
        
        print("=" * 80)
        print(f"{'MODEL CAPABILITIES':^80}")
        print("=" * 80)
        
        for model in models['modelSummaries']:
            print(f"\nModel: {model.get('modelId', 'N/A')}")
            print(f"Provider: {model.get('providerName', 'N/A')}")
            
            # Capabilities analysis
            input_modes = model.get('inputModalities', [])
            output_modes = model.get('outputModalities', [])
            
            print("Capabilities:")
            if 'TEXT' in input_modes:
                print("- Text Input ✓")
            if 'IMAGE' in input_modes:
                print("- Image Input ✓")
            if 'TEXT' in output_modes:
                print("- Text Generation ✓")
            if 'IMAGE' in output_modes:
                print("- Image Generation ✓")
    
    except Exception as e:
        print(f"Error listing model capabilities: {e}")

# Main execution
if __name__ == "__main__":
    # Choose which function to run
    print("Choose an option:")
    print("1. Detailed Model List")
    print("2. Model Capabilities")
    
    choice = input("Enter your choice (1/2): ").strip()
    
    if choice == '1':
        list_available_models()
    elif choice == '2':
        list_model_capabilities()
    else:
        print("Invalid choice. Please run the script again and select 1 or 2.")