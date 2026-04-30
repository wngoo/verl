# Git Push 403 / "send-pack: unexpected disconnect" 문제 해결

`main` 브랜치에서 다른 원격 레포로부터 pull 받은 뒤 GitHub에 push할 때 자주 마주치는 에러와 그 진단/해결 흐름 정리.

## 증상

```
error: RPC failed; HTTP 403 curl 22 The requested URL returned error: 403
send-pack: unexpected disconnect while reading sideband packet
fatal: the remote end hung up unexpectedly
Everything up-to-date
```

특징:
- **다른 브랜치는 push 정상**
- **main만, 그것도 다른 remote에서 pull 받은 직후 실패**
- `git branch -vv` 에서 `[origin/main: ahead 113]` 처럼 **앞선 커밋 수가 많음**
- `git remote -v` URL은 `https://github.com/...` 정상

## 원인 진단 흐름

### 1단계: 정말 푸시할 게 있는지 확인

`Everything up-to-date` 메시지에 속지 말 것. `ahead N`이 보이면 푸시 대상이 분명히 있는 상태다. 그 메시지는 다른 refspec(태그 등) 처리 결과이거나 부분 출력일 수 있다.

```bash
git branch -vv
git log --oneline origin/main..main | wc -l
```

여기서 `origin/main..main` 은 git의 **두 점 범위 문법**:
- `A..B` = "B에서는 도달 가능하지만 A에서는 도달 불가능한 커밋"
- 즉 "푸시 안 된 로컬 커밋만"
- `origin/main` 은 원격 추적 브랜치, `main` 은 로컬 브랜치 (서로 다른 ref)

### 2단계: 서버 측 거부인지 전송 계층 문제인지 분리

```bash
GIT_CURL_VERBOSE=1 git push origin main 2>&1 | grep -E "remote:|GH0|secret|exceed|protect"
```

- `remote:` 줄에 메시지가 **있으면** → 서버 정책 거부
  - `GH001: Large files` → 100MB 초과 파일
  - `secret detected` → push protection
  - `Protected branch update failed` → 브랜치 보호 규칙
- `remote:` 줄이 **전혀 없으면** → 전송 계층(transport) 문제
  - 서버가 응답을 보내기 전에 연결이 끊긴 것
  - 거의 항상 **푸시 페이로드 크기** 또는 **프록시/방화벽** 문제

## 해결 (전송 계층 문제일 때)

### 방법 1: SSH로 전환 (가장 깔끔, 가장 추천)

HTTPS 레이어(버퍼, 프록시, WAF, postBuffer 한도)를 전부 우회.

```bash
git remote set-url origin git@github.com:USER/REPO.git
ssh -T git@github.com           # "Hi USER!" 출력되면 OK
git push origin main
```

SSH 키가 없으면:
```bash
ssh-keygen -t ed25519 -C "your_email"
cat ~/.ssh/id_ed25519.pub
# 출력을 GitHub → Settings → SSH and GPG keys 에 등록
```

### 방법 2: HTTP postBuffer 늘리기

git의 기본 `http.postBuffer` 가 1MB라서 큰 푸시에서 끊긴다.

```bash
git config --global http.postBuffer 524288000   # 500MB
git config --get http.postBuffer                # 적용 확인
git push origin main
```

### 방법 3: 분할 푸시 (HTTPS 유지해야 할 때)

큰 푸시를 작은 조각으로 쪼개 점진적으로 올린다.

#### 3-1. 한 커밋씩

```bash
# 안 올라간 커밋 중 가장 오래된 것
git log --reverse --format=%H origin/main..main | head -1

# 그 SHA를 main으로 푸시
git push origin <SHA>:main
```

`<SHA>:main` 은 **로컬의 SHA를 원격의 main 브랜치로 push** 하라는 refspec.
- `<SHA>:main` 과 `<SHA>:refs/heads/main` 은 동일 (후자는 풀네임)
- `refs/heads/<name>` = 브랜치 네임스페이스 풀네임
- 원격에 main이 이미 있으면 짧은 형태로 충분

#### 3-2. N개씩 점프하며 자동 푸시

```bash
git log --reverse --format=%H origin/main..main > /tmp/todo.txt
wc -l /tmp/todo.txt

# 10개마다 한 번씩 push
awk 'NR%10==0' /tmp/todo.txt | while read sha; do
  echo "pushing $sha"
  git push origin "$sha:main" || break
done
git push origin main   # 나머지
```

#### 3-3. 특정 커밋에서만 실패하면 → 거대 blob 의심

해당 SHA가 추가한 파일들의 크기 확인:

```bash
git diff-tree --no-commit-id -r <FAILED_SHA> \
  | awk '$5=="A" || $5=="M" {print $4, $6}' \
  | while read blob path; do
      size=$(git cat-file -s "$blob")
      echo "$size $path"
    done | sort -n | tail
```

100MB 초과 파일이 있으면 GitHub HTTPS 푸시 하드 리밋에 걸린 것. `git filter-repo` 로 히스토리에서 제거하거나 Git LFS로 이전해야 한다.

### 방법 4: 프록시 환경 확인 (학교/사내망)

```bash
env | grep -i proxy
git config --get-all http.proxy
git config --get-all https.proxy
```

VPN이나 캠퍼스 네트워크가 큰 POST 요청을 끊는 경우가 있다. SSH로 전환(방법 1)이 가장 빠른 우회.

## 해결 (서버 정책 거부일 때 — `remote:` 줄에 메시지 있는 경우)

### 100MB 초과 파일

히스토리에서 큰 blob 찾기:

```bash
git rev-list --objects origin/main..main \
  | git cat-file --batch-check='%(objecttype) %(objectsize) %(rest)' \
  | awk '$1=="blob" && $2>50000000' \
  | sort -k2 -n
```

`git filter-repo --strip-blobs-bigger-than 100M` 또는 BFG로 정리.

### 시크릿 푸시 보호

`remote:` 메시지의 unblock URL을 따라가 GitHub UI에서 false positive 처리하거나, 히스토리에서 시크릿을 제거 후 force push (단독 작업 브랜치일 때만).

### 브랜치 보호 규칙

GitHub 레포 → Settings → Branches → main 규칙 확인. 직접 push가 막혀 있다면 PR을 통해 머지해야 한다.

## 추천 진단 순서

1. `git branch -vv` 로 ahead 수 확인
2. `GIT_CURL_VERBOSE=1 git push 2>&1 | grep remote:` 로 서버 메시지 유무 확인
3. `remote:` 메시지가 없으면 → **SSH 전환** (방법 1) 부터 시도
4. SSH 안 되면 → `http.postBuffer` (방법 2)
5. 그래도 안 되면 → 분할 푸시 (방법 3)
6. 특정 SHA에서만 막히면 → 거대 blob 확인

## 핵심 교훈

- `Everything up-to-date` + 403 + sideband 끊김 = **전송 계층 문제** (서버 정책 아님)
- 다른 브랜치는 되고 main만 안 되면 = **푸시 페이로드 크기** 문제일 가능성 매우 높음
- HTTPS에서 막힐 때 SSH 전환은 5분 안에 끝나는 가장 확실한 해결책
- `<SHA>:main` refspec으로 분할 푸시 가능 — 막히는 지점이 곧 거대 blob의 위치
