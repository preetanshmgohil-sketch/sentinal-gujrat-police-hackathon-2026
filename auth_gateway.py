import requests, json, warnings
warnings.filterwarnings('ignore')

COOKIE_VAL = 'eyJ1aWQiOiJkOGFjODVmMWJmYjhlN2VmIiwic2lkIjoiYWY3NmU2ZjY2OTY1ZmNhNTNkIn0.5ykSXMftAhv9SeZHUNvhJ8MoObRIvoWIGmI1GdHuXLo'
s = requests.Session()
s.verify = False
s.cookies.set('sentinel', COOKIE_VAL, domain='cctv.corp8.cloud')
s.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36',
    'Referer': 'https://cctv.corp8.cloud/',
    'Origin': 'https://cctv.corp8.cloud',
})

# Test with browser headers
print("=== Testing cam01 with browser headers ===", flush=True)
r = s.get('https://cctv.corp8.cloud/cam01/index.m3u8', timeout=20, allow_redirects=False)
print(f'Status: {r.status_code} CT:{r.headers.get("Content-Type","?")}', flush=True)
if r.status_code == 200:
    print(f'M3U8 content:\n{r.text[:800]}', flush=True)
else:
    print(f'Body: {r.text[:200]}', flush=True)

# Try with Accept header
s.headers.update({'Accept': '*/*'})
print("\n=== Testing with Accept */*", flush=True)
r2 = s.get('https://cctv.corp8.cloud/cam01/index.m3u8', timeout=20, allow_redirects=False)
print(f'Status: {r2.status_code} CT:{r2.headers.get("Content-Type","?")}', flush=True)
if r2.status_code == 200:
    print(f'Body: {r2.text[:500]}', flush=True)
else:
    print(f'Body: {r2.text[:200]}', flush=True)

# Try cam18, cam19, cam28 (the ones that were active in our grid)
print("\n=== Testing active cameras ===", flush=True)
for cid in ['cam18', 'cam19', 'cam28']:
    r3 = s.get(f'https://cctv.corp8.cloud/{cid}/index.m3u8', timeout=20, allow_redirects=False)
    print(f'  {cid}: {r3.status_code} CT:{r3.headers.get("Content-Type","?")}', flush=True)
    if r3.status_code == 200:
        print(f'    M3U8: {r3.text[:300]}', flush=True)
    else:
        print(f'    Body: {r3.text[:100]}', flush=True)
