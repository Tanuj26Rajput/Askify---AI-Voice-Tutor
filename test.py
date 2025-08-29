from murf import MurfDub
import os
murf_client = MurfDub(api_key=os.getenv("MURFDUB_API_KEY"))
print(murf_client.dubbing.locales.list())
