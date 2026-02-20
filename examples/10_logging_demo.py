"""Example showing structured logging and tracing capabilities."""

import logging
import time

from themis.utils import logging_utils, tracing

# Configure logging to output JSON to stderr
logging_utils.configure_logging(level="info", log_format="json")

logger = logging.getLogger("demo_app")


def process_item(item_id: int):
    with tracing.span("process_item", item_id=item_id):
        logger.info("Processing item", extra={"item_id": item_id, "status": "started"})
        time.sleep(0.1)
        if item_id % 3 == 0:
            logger.warning(
                "Item requires special handling",
                extra={"item_id": item_id, "reason": "multiple_of_3"},
            )
        logger.info("Item processed", extra={"item_id": item_id, "status": "completed"})


def main():
    # Enable tracing globally
    tracing.enable()

    logger.info("Starting demo application", extra={"version": "1.0.0"})

    with tracing.span("main_loop"):
        for i in range(1, 4):
            process_item(i)

    logger.info("Demo completed", extra={"total_items": 3})


if __name__ == "__main__":
    main()
