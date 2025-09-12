from dataclasses import dataclass


@dataclass
class WhatIsMyTextualRecommendationsQuery:
    text: str
    project_name: str
    model_version: str