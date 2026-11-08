"""One-time PriorLabs sign-in for the licence-gated TabPFN v3.5 weights.

Run in an interactive terminal. It opens the browser to ux.priorlabs.ai; after you sign in,
tabpfn verifies the licence acceptance and caches the API key in
~/.cache/tabpfn/auth_token, so later (non-interactive) runs such as
``python rerun_tabpfn.py v3.5`` work without a browser.

Workaround for tabpfn 9.2 on Windows: its login helper polls stdin with select.select(),
which only supports sockets on Windows (OSError WinError 10038). We replace only that
polling step with a plain wait for the browser callback; token verification, licence
check and caching are tabpfn's own code.
"""

from tabpfn import browser_auth


def _wait_for_browser_callback(auth_event, received_token, timeout_s=900):
    auth_event.wait(timeout_s)
    return received_token[0]


browser_auth._poll_for_token = _wait_for_browser_callback
browser_auth.ensure_license_accepted(hf_repo_id="tabpfn_3_5")
print("TabPFN v3.5 licence verified; API key cached in ~/.cache/tabpfn/auth_token")
