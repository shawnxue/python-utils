def testVariableInRec():
  def recMethod(lev):
    print(f"level_in_rec_before={lev}")
    print(f"check={check}")
    lev += 1
    print(f"level_in_rec_after={lev}")
    print(f"check={check}")

  level = 10
  check = 5
  print(f"level_outside_rec_before={level}")
  recMethod(level)
  print(f"level_outside_rec_after={level}")
  return

testVariableInRec()