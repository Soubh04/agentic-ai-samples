"""Test fixture: an MCP "server" that fails at startup, like a script with a missing dependency."""

import module_that_does_not_exist  # noqa: F401
