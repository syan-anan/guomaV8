import json, os, sys
p = r"H:\qinglong\脚本导出har\韵达.har"
with open(p, "r", encoding="utf-8") as f:
    har = json.load(f)
entries = har["log"]["entries"]

# 提取177请求的完整信息
e = entries[177]
req = e["request"]
print("=== [177] getImageVerifyCode 完整请求 ===")
print("Method:", req["method"])
print("URL:", req["url"])
print("\nHeaders:")
for h in req["headers"]:
    print(f"  {h['name']}: {h['value']}")
print("\nPostData:", req.get("postData", {}).get("text", ""))

print("\n\n=== 响应体(含图片) ===")
resp = e["response"]
print("Status:", resp["status"])
print("Content-Type:", resp["content"].get("mimeType"))
