# src/api/middleware/profiler.py
"""
Profiler Middleware
===================
Profile API requests using PyInstrument.
"""
from fastapi import Request, Response
from pyinstrument import Profiler
from pyinstrument.renderers.html import HTMLRenderer

async def profiler_middleware(request: Request, call_next):
    """
    Middleware to profile requests if `profile=true` query param is present.
    """
    profile_request = request.query_params.get("profile", "false").lower() == "true"
    
    if profile_request:
        profiler = Profiler(interval=0.001, async_mode="enabled")
        profiler.start()
        
        response = await call_next(request)
        
        profiler.stop()
        
        # If response is HTML, append profile? 
        # Or return profile as a separate endpoint?
        # For API, usually we want to see the profile.
        # Let's return the profile HTML if it was requested, overriding the JSON response.
        # This is for debugging only.
        
        html = profiler.output_text(unicode=True, color=True)
        # For browser view:
        html_report = profiler.output_html()
        
        return Response(
            content=html_report,
            media_type="text/html"
        )
        
    return await call_next(request)
