import requests
import pandas as pd
from datetime import datetime
from typing import List, Dict, Any, Union


class WeatherAPI:
    """
    A class to interact with the Open-Meteo Archive API for historical weather data.
    """
    
    BASE_URL = "https://archive-api.open-meteo.com/v1/archive"
    
    def __init__(self, latitude: float = -27.2142, longitude: float = -49.6431):
        """
        Initialize the WeatherAPI with default or custom coordinates.
        
        Args:
            latitude: The latitude of the location (default: -27.2142)
            longitude: The longitude of the location (default: -49.6431)
        """
        self.latitude = latitude
        self.longitude = longitude
    
    def validate_date(self, date_str: str) -> bool:
        """
        Validate if a string is in the correct date format (YYYY-MM-DD).
        
        Args:
            date_str: The date string to validate
            
        Returns:
            bool: True if valid, False otherwise
        """
        try:
            datetime.strptime(date_str, "%Y-%m-%d")
            return True
        except ValueError:
            return False
    
    def get_weather_data(
        self,
        start_date: str,
        end_date: str,
        hourly_metrics: Union[List[str], str] = None,
        daily_metrics: Union[List[str], str] = None,
        timezone: str = "America/Sao_Paulo"
    ) -> Dict[str, Any]:
        """
        Fetch weather data from the Open-Meteo Archive API.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            hourly_metrics: List of hourly metrics to retrieve or comma-separated string
                           (e.g., ["temperature_2m", "rain"] or "temperature_2m,rain")
            daily_metrics: List of daily metrics to retrieve or comma-separated string
            timezone: Timezone for the data (default: "GMT")
            
        Returns:
            Dict containing the API response
            
        Raises:
            ValueError: If dates are invalid or no metrics are provided
        """
        # Validate dates
        if not self.validate_date(start_date) or not self.validate_date(end_date):
            raise ValueError("Dates must be in format YYYY-MM-DD")
        
        # Check if at least one metric type is provided
        if not hourly_metrics and not daily_metrics:
            raise ValueError("At least one hourly or daily metric must be provided")
        
        # Prepare parameters
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "timezone": timezone
        }
        
        # Process hourly metrics
        if hourly_metrics:
            if isinstance(hourly_metrics, list):
                params["hourly"] = ",".join(hourly_metrics)
            else:
                params["hourly"] = hourly_metrics
        
        # Process daily metrics
        if daily_metrics:
            if isinstance(daily_metrics, list):
                params["daily"] = ",".join(daily_metrics)
            else:
                params["daily"] = daily_metrics
        
        # Make the API request
        response = requests.get(self.BASE_URL, params=params)
        
        # Check if request was successful
        if response.status_code == 200:
            return response.json()
        else:
            response.raise_for_status()
    
    def _json_to_dataframe(self, data: Dict[str, Any], data_type: str = "hourly") -> pd.DataFrame:
        """
        Convert API JSON response to pandas DataFrame.
        
        Args:
            data: API response JSON data
            data_type: Type of data to convert ('hourly' or 'daily')
            
        Returns:
            pandas DataFrame with the data
            
        Raises:
            KeyError: If the specified data_type is not found in the response
        """
        if data_type not in data:
            raise KeyError(f"No {data_type} data found in the API response")
        
        # Create DataFrame with time as index
        df = pd.DataFrame(data[data_type])
        
        # Convert time column to datetime and set as index
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
        return df
    
    def get_weather_data_as_df(
        self,
        start_date: str,
        end_date: str,
        hourly_metrics: Union[List[str], str] = None,
        timezone: str = "America/Sao_Paulo"
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch weather data and return as pandas DataFrame(s).
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            hourly_metrics: List of hourly metrics to retrieve or comma-separated string
            daily_metrics: List of daily metrics to retrieve or comma-separated string
            timezone: Timezone for the data (default: "GMT")
            
        Returns:
            Dict with keys 'hourly' and/or 'daily' containing pandas DataFrames
        """
        # Get JSON data
        json_data = self.get_weather_data(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=hourly_metrics,
            timezone=timezone
        )
        
        # Initialize result dict
        result = {}
        
        # Convert hourly data if present
        if 'hourly' in json_data:
            result['hourly'] = self._json_to_dataframe(json_data, 'hourly')
        
        # Convert daily data if present
        if 'daily' in json_data:
            result['daily'] = self._json_to_dataframe(json_data, 'daily')
        
        return result
    
    def get_temperature_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Convenience method to get temperature data.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            Dict containing temperature data
        """
        return self.get_weather_data(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["temperature_2m", "apparent_temperature"]
        )
    
    def get_temperature_data_as_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Convenience method to get temperature data as DataFrame.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing temperature data
        """
        data = self.get_weather_data_as_df(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["temperature_2m", "apparent_temperature"]
        )
        return data['hourly']
    
    def get_rain_data(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Convenience method to get rain data.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            Dict containing rain data
        """
        return self.get_weather_data(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["rain"]
        )
    
    def get_rain_data_as_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Convenience method to get rain data as DataFrame.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing rain data
        """
        data = self.get_weather_data_as_df(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["rain"]
        )
        return data['hourly']
    
    def get_all_metrics(self, start_date: str, end_date: str) -> Dict[str, Any]:
        """
        Get all standard weather metrics (temperature, humidity, apparent temperature, rain).
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            Dict containing all standard weather metrics
        """
        return self.get_weather_data(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain"]
        )
    
    def get_all_metrics_as_df(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Get all standard weather metrics as DataFrame.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing all standard weather metrics
        """
        data = self.get_weather_data_as_df(
            start_date=start_date,
            end_date=end_date,
            hourly_metrics=["temperature_2m", "relative_humidity_2m", "apparent_temperature", "rain"]
        )
        return data['hourly']
    
    def set_location(self, latitude: float, longitude: float) -> None:
        """
        Update the location coordinates for subsequent API calls.
        
        Args:
            latitude: New latitude value
            longitude: New longitude value
        """
        self.latitude = latitude
        self.longitude = longitude


# Example usage
if __name__ == "__main__":
    # Create API instance
    weather_api = WeatherAPI()
    
    # Get data for the last week as DataFrame
    df = weather_api.get_all_metrics_as_df(
        start_date="2023-01-01",
        end_date="2023-01-07"
    )
    
    # Print DataFrame info
    print(f"DataFrame shape: {df.shape}")
    print("\nDataFrame head:")
    print(df.head())
