import asyncio
from bleakheart import PolarH10

class BioSensor:
    def __init__(self):
        self.polar = PolarH10()
        self.heart_rate = 0

    async def connect(self):
        """Connect to the Polar H10 sensor."""
        await self.polar.connect()

    async def stream_hr(self):
        """Continuously stream heart rate data."""
        async for sample in self.polar.heart_rate():
            self.heart_rate = sample.bpm
            print(f"Heart Rate: {self.heart_rate} BPM")

    async def disconnect(self):
        """Disconnect from the sensor."""
        await self.polar.disconnect()

async def main():
    sensor = BioSensor()
    await sensor.connect()
    await sensor.stream_hr()

if __name__ == "__main__":
    asyncio.run(main())

