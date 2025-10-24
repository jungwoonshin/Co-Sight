#!/usr/bin/env python3
"""
Test script to verify Google Search API configuration
"""

import os
import requests
import json
from typing import Dict, Any, List

def test_google_search_api(api_key: str, search_engine_id: str, test_query: str = "artificial intelligence") -> Dict[str, Any]:
    """
    Test Google Custom Search API with provided credentials
    
    Args:
        api_key: Google API key
        search_engine_id: Google Custom Search Engine ID
        test_query: Test search query
    
    Returns:
        Dictionary with test results
    """
    
    # Google Custom Search API endpoint
    url = "https://www.googleapis.com/customsearch/v1"
    
    # Parameters for the API call
    params = {
        'key': api_key,
        'cx': search_engine_id,
        'q': test_query,
        'num': 3  # Number of results to return
    }
    
    print(f"🔍 Testing Google Search API...")
    print(f"   API Key: {api_key[:10]}...{api_key[-4:]}")
    print(f"   Search Engine ID: {search_engine_id}")
    print(f"   Test Query: '{test_query}'")
    print(f"   URL: {url}")
    print()
    
    try:
        # Make the API request
        response = requests.get(url, params=params, timeout=10)
        
        # Check HTTP status
        if response.status_code == 200:
            data = response.json()
            
            # Extract search results
            search_info = data.get('searchInformation', {})
            items = data.get('items', [])
            
            print("✅ SUCCESS: Google Search API is working!")
            print(f"   Total Results: {search_info.get('totalResults', 'Unknown')}")
            print(f"   Search Time: {search_info.get('searchTime', 'Unknown')} seconds")
            print(f"   Results Returned: {len(items)}")
            print()
            
            # Display first few results
            if items:
                print("📋 Sample Results:")
                for i, item in enumerate(items[:3], 1):
                    print(f"   {i}. {item.get('title', 'No title')}")
                    print(f"      URL: {item.get('link', 'No URL')}")
                    print(f"      Snippet: {item.get('snippet', 'No snippet')[:100]}...")
                    print()
            
            return {
                'status': 'success',
                'status_code': response.status_code,
                'total_results': search_info.get('totalResults'),
                'search_time': search_info.get('searchTime'),
                'results_count': len(items),
                'sample_results': items[:3] if items else []
            }
            
        else:
            error_data = response.json() if response.content else {}
            error_message = error_data.get('error', {}).get('message', 'Unknown error')
            
            print(f"❌ ERROR: API request failed")
            print(f"   Status Code: {response.status_code}")
            print(f"   Error Message: {error_message}")
            print()
            
            return {
                'status': 'error',
                'status_code': response.status_code,
                'error_message': error_message
            }
            
    except requests.exceptions.Timeout:
        print("❌ ERROR: Request timed out")
        return {'status': 'timeout', 'error': 'Request timed out'}
        
    except requests.exceptions.RequestException as e:
        print(f"❌ ERROR: Request failed - {str(e)}")
        return {'status': 'request_error', 'error': str(e)}
        
    except Exception as e:
        print(f"❌ ERROR: Unexpected error - {str(e)}")
        return {'status': 'unexpected_error', 'error': str(e)}


def test_with_environment_variables():
    """Test using environment variables"""
    print("🔧 Testing with environment variables...")
    
    api_key = os.getenv("GOOGLE_API_KEY")
    search_engine_id = os.getenv("SEARCH_ENGINE_ID")
    
    if not api_key or not search_engine_id:
        print("❌ Missing environment variables:")
        print(f"   GOOGLE_API_KEY: {'✅ Set' if api_key else '❌ Missing'}")
        print(f"   SEARCH_ENGINE_ID: {'✅ Set' if search_engine_id else '❌ Missing'}")
        return None
    
    return test_google_search_api(api_key, search_engine_id)


def main():
    """Main test function"""
    print("=" * 60)
    print("🧪 Google Search API Test")
    print("=" * 60)
    print()
    
    # Test with provided credentials
    api_key = "AIzaSyBkF5LRZ8Fc9m6spjH3WGRjsioK3IutrKc"
    search_engine_id = "c00ea11fbeff6417d"
    
    print("1️⃣ Testing with provided credentials:")
    result1 = test_google_search_api(api_key, search_engine_id)
    print()
    
    # Test with environment variables (if they exist)
    print("2️⃣ Testing with environment variables:")
    result2 = test_with_environment_variables()
    print()
    
    # Summary
    print("=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    
    if result1 and result1['status'] == 'success':
        print("✅ Provided credentials: WORKING")
        print(f"   - Found {result1['results_count']} results")
        print(f"   - Search time: {result1['search_time']} seconds")
    else:
        print("❌ Provided credentials: FAILED")
        if result1:
            print(f"   - Error: {result1.get('error_message', result1.get('error', 'Unknown'))}")
    
    if result2 and result2['status'] == 'success':
        print("✅ Environment variables: WORKING")
        print(f"   - Found {result2['results_count']} results")
        print(f"   - Search time: {result2['search_time']} seconds")
    elif result2 is None:
        print("⚠️  Environment variables: NOT SET")
    else:
        print("❌ Environment variables: FAILED")
        if result2:
            print(f"   - Error: {result2.get('error_message', result2.get('error', 'Unknown'))}")
    
    print()
    print("💡 Next steps:")
    if result1 and result1['status'] == 'success':
        print("   1. Create a .env file with your credentials")
        print("   2. Add: GOOGLE_API_KEY=AIzaSyBkF5LRZ8Fc9m6spjH3WGRjsioK3IutrKc")
        print("   3. Add: SEARCH_ENGINE_ID=c00ea11fbeff6417d")
        print("   4. Restart your Co-Sight application")
    else:
        print("   1. Check your Google API key and Search Engine ID")
        print("   2. Verify the Custom Search API is enabled")
        print("   3. Check your API quotas and billing")


if __name__ == "__main__":
    main()
