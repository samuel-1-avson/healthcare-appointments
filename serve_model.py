#!/usr/bin/env python
"""
Model Serving CLI
=================
Command-line interface for starting the prediction API server.

Usage:
    python serve_model.py                      # Start with defaults
    python serve_model.py --port 8080          # Custom port
    python serve_model.py --reload             # Development mode
    python serve_model.py --workers 4          # Production mode

Examples:
    # Development (with auto-reload)
    python serve_model.py --reload --debug
    
    # Production
    python serve_model.py --host 0.0.0.0 --port 8000 --workers 4
"""

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Start the Healthcare No-Show Prediction API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python serve_model.py                    # Default settings
  python serve_model.py --reload           # Development with auto-reload
  python serve_model.py --port 8080        # Custom port
  python serve_model.py --workers 4        # Multiple workers (production)
        """
    )
    
    # Server settings
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Server host (default: 0.0.0.0)"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Server port (default: 8000)"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of worker processes (default: 1)"
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development mode)"
    )
    
    # Model settings
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Path to model file (overrides config)"
    )
    parser.add_argument(
        "--preprocessor",
        type=str,
        default=None,
        help="Path to preprocessor file (overrides config)"
    )
    
    # Other settings
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    logger = logging.getLogger("serve_model")
    
    # Set environment variables for overrides
    import os
    if args.model:
        os.environ["NOSHOW_MODEL_PATH"] = args.model
    if args.preprocessor:
        os.environ["NOSHOW_PREPROCESSOR_PATH"] = args.preprocessor
    if args.debug:
        os.environ["NOSHOW_DEBUG"] = "true"
    
    # Start server
    logger.info("=" * 50)
    logger.info("Healthcare No-Show Prediction API")
    logger.info("=" * 50)
    logger.info(f"Host: {args.host}")
    logger.info(f"Port: {args.port}")
    logger.info(f"Workers: {args.workers}")
    logger.info(f"Reload: {args.reload}")
    logger.info(f"Debug: {args.debug}")
    logger.info("=" * 50)
    
    try:
        import uvicorn
        
        uvicorn.run(
            "src.api.main:app",
            host=args.host,
            port=args.port,
            reload=args.reload,
            workers=args.workers if not args.reload else 1,
            log_level=args.log_level.lower()
        )
        
    except KeyboardInterrupt:
        logger.info("Server stopped by user")
    except Exception as e:
        logger.error(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()