FROM python:3.12.9-bookworm

WORKDIR /usr/src/transcription-pipeline

COPY . .

RUN pip install --no-cache-dir -r requirements-initial.txt
RUN pip install --no-cache-dir -e .
RUN pip install --no-cache-dir -r requirements-adjustments.txt

CMD [ "transcription-processor", "--domain", "my.apartmentlines.com", "--limit", "1", "--processing-limit", "2", "--debug" ]
