import pandas as pd
from pathlib import Path
from typing import Literal, Optional
from loguru import logger

from .api import WeatherAPI
from .scraping import WebScraper

class DataGenerator:
    """
    A class that generates datasets for training ML models or making predictions.
    Combines weather data from API and water level data from web scraping.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """
        Initialize the DataGenerator.
        
        Args:
            output_dir: Directory to save generated datasets (default: data/output)
        """
        self.weather_api = WeatherAPI()
        self.scraper = WebScraper()
        
        if output_dir is None:
            self.output_dir = Path(__file__).parent / "output"
        else:
            self.output_dir = Path(output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"DataGenerator initialized with output directory: {self.output_dir}")
    
    def _get_weather_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch weather data from the API.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing weather data
        """
        logger.info(f"Fetching weather data from {start_date} to {end_date}")
        try:
            # Get all relevant weather metrics
            df = self.weather_api.get_all_metrics_as_df(start_date, end_date)
            logger.info(f"Weather data fetched successfully: {df.shape} rows")
            return df
        except Exception as e:
            logger.error(f"Error fetching weather data: {str(e)}")
            raise
    
    def _get_water_level_data(self, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch water level data via web scraping.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            
        Returns:
            DataFrame containing water level data
        """
        logger.info(f"Scraping water level data from {start_date} to {end_date}")
        try:
            df = self.scraper.parse_data(start_date, end_date)
            logger.info(f"Water level data scraped successfully: {df.shape} rows")
            return df
        except Exception as e:
            logger.error(f"Error scraping water level data: {str(e)}")
            raise
    
    def _merge_datasets(self, weather_df: pd.DataFrame, level_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Merge weather and water level data by timestamp.
        
        Args:
            weather_df: DataFrame with weather data
            level_df: DataFrame with water level data (optional)
            
        Returns:
            Merged DataFrame
        """
        # If no level data provided, return weather data only (for prediction)
        if level_df is None:
            return weather_df
        
        # Ensure index is datetime for both datasets
        if not isinstance(weather_df.index, pd.DatetimeIndex):
            weather_df.set_index('time', inplace=True)
        
        # Rename column for clarity
        level_df = level_df.rename(columns={"Nível": "water_level"})
        
        # Resample and merge datasets (joining on datetime index)
        # Use outer join and then fill missing values
        merged_df = pd.merge_asof(
            weather_df.reset_index(),
            level_df.rename(columns={"Data": "time"}).sort_values("time"),
            on="time",
            direction="nearest",
            tolerance=pd.Timedelta("1h")
        )
        
        logger.info(f"Datasets merged successfully: {merged_df.shape} rows")
        return merged_df
    
    def _process_data(self, df: pd.DataFrame, type: Literal["train", "predict"]) -> pd.DataFrame:
        """
        Process the dataset for training or prediction.
        
        Args:
            df: DataFrame with merged data
            type: Whether processing for "train" or "predict"
            
        Returns:
            Processed DataFrame
        """
        # Handle missing values
        df = df.interpolate(method='linear')
        
        # Add engineered features
        df['hour'] = df['time'].dt.hour
        df['day_of_week'] = df['time'].dt.dayofweek
        df['month'] = df['time'].dt.month
        
        # Calculate rolling averages for weather metrics
        df['rain_24h'] = df['rain'].rolling(window=24).sum()
        df['temperature_24h_avg'] = df['temperature_2m'].rolling(window=24).mean()
        df['humidity_24h_avg'] = df['relative_humidity_2m'].rolling(window=24).mean()
        
        # Drop rows with NaN values (from rolling windows)
        df = df.dropna()
        
        # For training data, include the water level and future water levels
        if type == "train" and "water_level" in df.columns:
            # Create target variables (water level in future hours)
            for hours in [1, 3, 6, 12, 24]:
                df[f'water_level_next_{hours}h'] = df['water_level'].shift(-hours)
            
            # Drop rows where future values are NaN
            df = df.dropna()
        
        logger.info(f"Data processing completed: {df.shape} rows")
        return df
    
    def generate(
        self, 
        start_date: str, 
        end_date: str, 
        type: Literal["train", "predict"] = "train",
        save: bool = True,
        filename: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Generate a dataset for training a machine learning model or for prediction.
        
        Args:
            start_date: Start date in format YYYY-MM-DD
            end_date: End date in format YYYY-MM-DD
            type: Dataset type - "train" (includes scraped water level data) 
                  or "predict" (only API weather data)
            save: Whether to save the dataset to a file
            filename: Custom filename (default: {type}_data_{start_date}_{end_date}.csv)
            
        Returns:
            DataFrame containing the generated dataset
        """
        logger.info(f"Generating {type} dataset from {start_date} to {end_date}")
        
        # Get weather data from API
        weather_df = self._get_weather_data(start_date, end_date)
        
        # For training data, also get water level from scraping
        if type == "train":
            level_df = self._get_water_level_data(start_date, end_date)
            merged_df = self._merge_datasets(weather_df, level_df)
        else:
            # For prediction, only use weather data
            merged_df = weather_df.reset_index()
            
        # Process data based on type
        final_df = self._process_data(merged_df, type)
        
        # Save to file if requested
        if save:
            if filename is None:
                filename = f"{type}_data_{start_date}_{end_date}.csv"
            
            file_path = self.output_dir / filename
            final_df.to_csv(file_path, index=False)
            logger.info(f"Dataset saved to {file_path}")
        
        return final_df


def generate(
    start_date: str, 
    end_date: str, 
    type: Literal["train", "predict"] = "train",
    output_dir: Optional[str] = None,
    save: bool = True,
    filename: Optional[str] = None
) -> pd.DataFrame:
    """
    Convenience function to generate a dataset without creating a DataGenerator instance.
    
    Args:
        start_date: Start date in format YYYY-MM-DD
        end_date: End date in format YYYY-MM-DD
        type: Dataset type - "train" (includes scraped water level data) 
              or "predict" (only API weather data)
        output_dir: Directory to save generated datasets
        save: Whether to save the dataset to a file
        filename: Custom filename
        
    Returns:
        DataFrame containing the generated dataset
    """
    generator = DataGenerator(output_dir=output_dir)
    return generator.generate(
        start_date=start_date,
        end_date=end_date,
        type=type,
        save=save,
        filename=filename
    )


if __name__ == "__main__":
    # Example usage
    train_data = generate(
        start_date="2023-01-01", 
        end_date="2023-12-31", 
        type="train"
    )
    
    predict_data = generate(
        start_date="2024-01-01",
        end_date="2024-01-31",
        type="predict"
    )
    
    print(f"Train data shape: {train_data.shape}")
    print(f"Predict data shape: {predict_data.shape}")
