with open('../dat/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5', encoding='utf-8', mode='r') as f:
    cnt = 0
    for line in f:
        cnt += 1
        if(cnt == 1000):
            break
        print(line)
