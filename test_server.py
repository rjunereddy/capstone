import urllib.request, json

def post(payload):
    data = json.dumps(payload).encode()
    req  = urllib.request.Request(
        'http://127.0.0.1:5000/analyze',
        data=data, headers={'Content-Type':'application/json'}, method='POST')
    return json.loads(urllib.request.urlopen(req, timeout=15).read())

def get(path):
    return json.loads(urllib.request.urlopen('http://127.0.0.1:5000'+path, timeout=5).read())

print('=== PhishGuard End-to-End Test Suite ===')
print()

h = get('/health')
print('HEALTH:', json.dumps(h, indent=2))
print()

tests = [
    ('https://www.google.com',               'Search the world information',                          False, 0.30),
    ('https://www.irctc.co.in',              'Indian Railways book train ticket',                    False, 0.30),
    ('https://www.hdfcbank.com',             'HDFC personal banking login',                          False, 0.30),
    ('http://paypal-secure-verify.tk/login', 'urgent verify your paypal account click here OTP',    True,  0.60),
    ('http://sbi-account-verify.xyz/secure', 'Dear Customer SBI account suspended verify otp kyc',  True,  0.60),
    ('http://192.168.1.1/banking/login',     'bank login username password',                         True,  0.50),
    ('http://bit.ly/xyzabc',                 'click here claim your prize winner lottery',           True,  0.45),
]

passed = 0
for url, text, expect_high, threshold in tests:
    d      = post({'url': url, 'text': text})
    score  = d['final_risk_score']
    method = d.get('url_method', '?')
    level  = d['risk_level']
    ok     = (score >= threshold) if expect_high else (score < threshold)
    if ok: passed += 1
    tag = 'PASS' if ok else 'WARN'
    print('[{}] {:54s} {:.0%}  [{}]  {}'.format(tag, url[:54], score, method, level))

print()
print('Results: {}/{}'.format(passed, len(tests)))
