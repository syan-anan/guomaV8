# -*- coding: utf-8 -*-
"""客户端重试模板 - 调用本服务 API, 失败自动换新验证码重试.\n用法: 传入 fetch_captcha 和 submit_answer 回调, 即可达到 98%+ 有效通过率.\n"""
import time


def solve_with_retry(fetch_captcha, submit_answer, solve_one, max_retry=3, pause=0.3):
    """
    fetch_captcha() -> captcha_data    拉取验证码(图片/flag 等)
    solve_one(captcha_data) -> answer  调 API 识别
    submit_answer(answer, captcha_data) -> bool  提交给目标网站, True=通过
    重试 max_retry 次, 返回 (ok, attempts)
    """
    for attempt in range(1, max_retry + 1):
        try:
            cap = fetch_captcha()
            ans = solve_one(cap)
            if submit_answer(ans, cap):
                return True, attempt
        except Exception as e:
            print("attempt", attempt, "error:", e)
        time.sleep(pause)
    return False, max_retry
