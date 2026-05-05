build:
	docker build -t salary-api .

run:
	docker run -p 8000:8000 salary-api
