UPPERCASE = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
LOWERCASE = UPPERCASE.lower()
DIGITS = "0123456789"
ALL = UPPERCASE + LOWERCASE + DIGITS
def random_password(*, upper, lower, digits, length):
  import random
  print(*(random.choice(UPPERCASE) for _ in range(upper)))
  chars = [
    *(random.choice(UPPERCASE) for _ in range(upper)),
    *(random.choice(LOWERCASE) for _ in range(lower)),
    *(random.choice(DIGITS) for _ in range(digits)),
    *(random.choice(ALL) for _ in range(length-upper-lower-digits)),
  ]
  random.shuffle(chars)
  return "".join(chars)

print(random_password(upper=5, lower=3, digits=4, length=15))