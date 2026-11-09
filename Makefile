.PHONY: build up down logs test

build:
	docker compose build

up:
	docker compose up -d --build

down:
	docker compose down

logs:
	docker compose logs -f

test:
	curl -s http://localhost:6090/sante | python3 -m json.tool
