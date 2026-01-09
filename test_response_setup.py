"""
Quick Test Script for Response Generation
Tests if groq package is installed and API key works
"""

print("\n" + "="*80)
print("TESTING RESPONSE GENERATION SETUP")
print("="*80 + "\n")

# Test 1: Check groq package
print("Test 1: Checking groq package installation...")
try:
    import groq
    print("   ✅ groq package installed successfully")
except ImportError as e:
    print(f"   ❌ Error: {e}")
    print("   Fix: pip install groq")
    exit(1)

# Test 2: Check API key
print("\nTest 2: Testing API connection...")
import os
api_key = os.getenv('GROQ_API_KEY')
if not api_key:
    api_key = input("Enter your Groq API key: ").strip()
if not api_key:
    print("   ❌ API key required")
    exit(1)

try:
    from groq import Groq
    client = Groq(api_key=api_key)
    print("   ✅ API client initialized")
    
    # Quick test
    print("\nTest 3: Generating sample response...")
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Say 'Hello, I'm working!' in 3 words."}
        ],
        model="llama-3.3-70b-versatile",
        max_tokens=20
    )
    
    response = chat_completion.choices[0].message.content
    print(f"   ✅ Response received: {response}")
    
except Exception as e:
    print(f"   ❌ Error: {e}")
    print("\n   Possible issues:")
    print("   1. No internet connection")
    print("   2. Invalid API key")
    print("   3. Groq API service down")
    exit(1)

print("\n" + "="*80)
print("✅ ALL TESTS PASSED! Response generation is ready.")
print("="*80)
print("\n✨ You can now run: python chatbot_pipeline.py")
print("\n")
