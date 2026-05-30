from server.tools.api.weather_tool import (
    get_current_weather,
    get_travel_advice,
    get_astronomy_info,
    get_weather_forecast,
)

from server.tools.api.map_tool import (
    get_walking_plan,
    get_distance,
    search_around,
    get_static_map,
)

from server.tools.api.time_tool import get_current_time
from server.tools.api.web_search_tool import web_search
from server.tools.api.image_generation_tool import generate_image_tool
from server.rag.rag_tool import search_knowledge_base

ALL_AVAILABLE_TOOLS = [
    get_current_weather,
    get_travel_advice,
    get_astronomy_info,
    get_weather_forecast,
    get_walking_plan,
    get_distance,
    search_around,
    get_static_map,
    get_current_time,
    web_search,
    generate_image_tool,
    search_knowledge_base,
]