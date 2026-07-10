import re

def get_chatbot_response(prompt, latest_data):
    """
    Generates a rule-based response based on user prompt and latest telemetry.
    Supports English and basic Tamil responses.
    """
    prompt = prompt.lower().strip()
    
    # Check if user requested Tamil output
    is_tamil = "tamil" in prompt or "தமிழ்" in prompt
    
    # Extract current data for context
    vcc = latest_data.get("vcc", "Unknown")
    heap = latest_data.get("heap", "Unknown")
    status = latest_data.get("risk_status", "UNKNOWN")
    prob = latest_data.get("failure_probability", "Unknown")
    
    # Rule 1: Voltage Inquiry
    if re.search(r'voltage|vcc', prompt):
        if is_tamil:
            return f"தற்போதைய மின்னழுத்தம் {vcc}V ஆக உள்ளது."
        return f"The current internal voltage (VCC) is {vcc}V."
        
    # Rule 2: Memory Inquiry
    elif re.search(r'heap|memory', prompt):
        if is_tamil:
            return f"தற்போதைய Heap memory {heap} bytes ஆக உள்ளது."
        return f"The current free heap memory is {heap} bytes."
        
    # Rule 3: Safety / Risk Status
    elif re.search(r'safe|status|risk', prompt) and not re.search(r'why', prompt):
        if is_tamil:
            if status == "SAFE":
                return "சாதனம் பாதுகாப்பாக உள்ளது (SAFE)."
            elif status == "WARNING":
                return "சாதனம் எச்சரிக்கை நிலையில் உள்ளது (WARNING)."
            else:
                return f"ஆபத்து! சாதனம் செயலிழக்க அதிக வாய்ப்புள்ளது ({prob}%)."
        else:
            return f"The device is currently in a {status} state with a failure probability of {prob}%."
            
    # Rule 4: "Why is risk high?"
    elif "why" in prompt and "high" in prompt:
        return "Risk is usually high when VCC fluctuates abnormally or free heap memory drops critically low, indicating a potential memory leak or hardware fault."
        
    # Rule 5: Memory Leak Fix
    elif "fix" in prompt and "leak" in prompt:
        return "To fix a memory leak, review your firmware code to ensure dynamically allocated memory is properly freed, avoid recursive loops that consume stack space, and optimize data structures."
        
    # Rule 6: Red LED Meaning
    elif "led" in prompt and "red" in prompt:
        return "A red LED typically indicates that the device has entered a DANGER risk status, meaning the failure probability is >70%. Check the voltage and memory levels immediately."
        
    # Default Fallback Responses
    if is_tamil:
        return "மன்னிக்கவும், எனக்கு புரியவில்லை. மின்னழுத்தம் அல்லது நினைவகம் பற்றி கேட்கவும்."
    return "I am a rule-based assistant. I can answer questions about current voltage, heap memory, risk status, memory leaks, or why risk might be high. Please rephrase your question."
