import torch

class MetaData:
    def __init__(
        self,
        base_lon=180,
        base_lat=90,
        # base_year=1980,
        max_gsd=10.,
        scale=1000
    ):
        self.base_lon = base_lon
        self.base_lat = base_lat
        # self.base_year = base_year
        self.max_gsd = max_gsd
        self.scale = scale
    
    def normalize(self, metadata):
        lon, lat, gsd, year = metadata
        lon = lon / (180 + self.base_lon) * self.scale
        lat = lat / (90 + self.base_lat) * self.scale
        gsd = gsd / self.max_gsd * self.scale
        year = year / 2100 * self.scale
        # month = month / 12 * self.scale
        # day = day / 31 * self.scale
        return torch.tensor([lon, lat, gsd, year])

    def unnormalize(self, metadata, is_print=False):
        lon, lat, gsd, year = metadata
        lon = lon / self.scale * (180 + self.base_lon) - (self.base_lon if is_print else 0)
        lat = lat / self.scale * (90 + self.base_lat) - (self.base_lat if is_print else 0)
        gsd = gsd / self.scale * self.max_gsd
        year = year / self.scale * 2100 + (self.base_year if is_print else 0)
        # month = month / self.scale * 12
        # day = day / self.scale * 31
        return torch.tensor([lon, lat, gsd, year])
    
    def text_formatting(self, metadata):
        lon, lat, gsd, year = metadata

        text_caption = f' at a resolution of {round(gsd, 5)}. The longitude, latitude is {lon}, {lat}.'
        if year != 0.:
            text_caption += f' The date is {year}.'
        
        return text_caption
    
    def embedding(self, metadata):
        # norm_data, text_caption
        return self.normalize(metadata), self.text_formatting(metadata)
        
