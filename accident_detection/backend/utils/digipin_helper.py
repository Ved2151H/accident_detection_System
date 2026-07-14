import logging
import digipin

logger = logging.getLogger("AegisEye.DigiPIN")

class DigiPinHelper:
    def __init__(self):
        self.dp = digipin.Digipin()

    def gps_to_digipin(self, lat: float, lon: float) -> str:
        """Convert GPS coordinates to DIGIPIN string.
        Falls back to default Pune coordinates if out of bounds or invalid.
        """
        try:
            lat = float(lat)
            lon = float(lon)
        except (ValueError, TypeError):
            logger.warning(f"Invalid GPS coordinate type: lat={lat}, lon={lon}. Defaulting to Pune central.")
            lat, lon = 18.5204, 73.8567

        # DIGIPIN bounds for India
        if not (2.5 <= lat <= 38.5) or not (63.5 <= lon <= 99.5):
            logger.warning(f"GPS Coordinates ({lat}, {lon}) are outside India DIGIPIN bounds. Falling back to Pune central.")
            lat, lon = 18.5204, 73.8567

        try:
            return self.dp.get_digipin(lat, lon)
        except Exception as e:
            logger.error(f"Error encoding coordinates ({lat}, {lon}) to DIGIPIN: {e}")
            return "4FP-492-CMTF"

    def digipin_to_gps(self, code: str) -> tuple:
        """Decode a DIGIPIN string to GPS coordinates.
        Returns Pune central coordinates on error.
        """
        if not code or not isinstance(code, str):
            return 18.5204, 73.8567
        try:
            clean_code = code.replace("-", "").replace(" ", "").upper()
            coords = self.dp.get_lat_lng_from_digipin(clean_code)
            return coords.latitude, coords.longitude
        except Exception as e:
            logger.error(f"Error decoding DIGIPIN code '{code}': {e}")
            return 18.5204, 73.8567
