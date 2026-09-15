#![no_std]

extern crate alloc;

use alloc::boxed::Box;
use alloc::sync::Arc;
use core::future::Future;
use core::pin::Pin;

use embassy_sync::blocking_mutex::raw::CriticalSectionRawMutex;
use embassy_sync::mutex::Mutex;
use xpanse_api::{
    bus::{
        allocator::BusAllocator,
        i2c::{I2cBusHandle, I2cError},
    },
    driver::{Driver, DriverError, DriverMeta},
    gpio_bank::{BankPins, GpioBank},
    interfaces::leds::{Generic, pin_led},
    metadata::{ModuleDetectResistor, ModuleID, ModuleSlot},
    reexports::embassy_time::Timer,
    registry::Registry,
};

/// Bit-banged I2C clock for the module bus.
const I2C_FREQUENCY_HZ: u32 = 100_000;

/// BME280 address with SDO tied to 3V3.
const BME280_ADDR: u8 = 0x77;
const BME280_CHIP_ID: u8 = 0xD0;
const BME280_CHIP_ID_VALUE: u8 = 0x60;
/// Temperature and pressure calibration, 0x88..=0xA1 (the last byte is dig_H1).
const BME280_CALIB_TP: u8 = 0x88;
/// Remaining humidity calibration, 0xE1..=0xE7.
const BME280_CALIB_H: u8 = 0xE1;
const BME280_CTRL_HUM: u8 = 0xF2;
const BME280_STATUS: u8 = 0xF3;
const BME280_CTRL_MEAS: u8 = 0xF4;
/// Pressure, temperature and humidity ADC values, 0xF7..=0xFE.
const BME280_DATA: u8 = 0xF7;
/// Humidity oversampling x1.
const BME280_CTRL_HUM_VALUE: u8 = 0b001;
/// Temperature and pressure oversampling x1 (osrs_t = osrs_p = 0b001), forced
/// mode (0b01, one conversion per write).
const BME280_CTRL_MEAS_FORCED: u8 = 0b0010_0101;
const BME280_STATUS_MEASURING: u8 = 0x08;
/// A x1/x1/x1 conversion takes under 10 ms, so 10 polls 2 ms apart is plenty.
const BME280_MAX_POLLS: u32 = 10;
const BME280_POLL_INTERVAL_MS: u64 = 2;

/// LIS3DH address with SDO/SA0 tied to 3V3.
const LIS3DH_ADDR: u8 = 0x19;
const LIS3DH_WHO_AM_I: u8 = 0x0F;
const LIS3DH_WHO_AM_I_VALUE: u8 = 0x33;
const LIS3DH_CTRL_REG1: u8 = 0x20;
const LIS3DH_CTRL_REG4: u8 = 0x23;
const LIS3DH_OUT_X_L: u8 = 0x28;
/// Setting the sub-address MSB makes the LIS3DH auto-increment on multi-byte reads.
const LIS3DH_AUTO_INCREMENT: u8 = 0x80;
/// 100 Hz output data rate, normal mode, X/Y/Z enabled.
const LIS3DH_CTRL_REG1_VALUE: u8 = 0x57;
/// Block data update, ±2 g full scale, high-resolution (12-bit) mode.
const LIS3DH_CTRL_REG4_VALUE: u8 = 0x88;
/// Sensitivity at ±2 g in high-resolution mode is 1 mg per digit.
const LIS3DH_G_PER_DIGIT: f32 = 0.001;

/// Error returned by the module's sensor capabilities.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SensorError {
    /// I2C communication with the sensor failed.
    BusError,
    /// The sensor answered with an unexpected chip ID.
    NoDevice,
    /// The sensor did not finish a measurement in time.
    Timeout,
    /// The sensor returned data that could not be processed.
    InvalidData,
}

impl From<I2cError> for SensorError {
    fn from(_: I2cError) -> Self {
        SensorError::BusError
    }
}

/// One environmental measurement from the BME280.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EnvReading {
    /// Temperature in degrees Celsius.
    pub temperature_c: f32,
    /// Relative humidity in percent.
    pub humidity_percent: f32,
    /// Barometric pressure in pascals.
    pub pressure_pa: f32,
}

/// Async interface for reading temperature, humidity and pressure.
///
/// Apps lease it from the registry as `Box<dyn EnvSensor>`.
pub trait EnvSensor: Send {
    /// Trigger a measurement and return the compensated result.
    fn read<'a>(&'a mut self)
    -> Pin<Box<dyn Future<Output = Result<EnvReading, SensorError>> + 'a>>;
}

/// Acceleration on each axis, in units of g.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Acceleration {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

/// Async interface for reading 3-axis acceleration.
///
/// Apps lease it from the registry as `Box<dyn Accelerometer>`.
pub trait Accelerometer: Send {
    /// Return the most recent acceleration sample.
    fn read<'a>(&'a mut self)
    -> Pin<Box<dyn Future<Output = Result<Acceleration, SensorError>> + 'a>>;
}

/// One device's handle to the I2C bus shared by both sensors.
///
/// Every call holds the bus lock for the whole transfer, so the sensors can be
/// leased by different apps without their transfers interleaving.
struct SharedI2c(Arc<Mutex<CriticalSectionRawMutex, I2cBusHandle>>);

impl SharedI2c {
    async fn write(&self, address: u8, bytes: &[u8]) -> Result<(), I2cError> {
        self.0.lock().await.write(address, bytes).await
    }

    async fn read_regs(&self, address: u8, reg: u8, buf: &mut [u8]) -> Result<(), I2cError> {
        self.0.lock().await.write_read(address, &[reg], buf).await
    }

    async fn read_reg(&self, address: u8, reg: u8) -> Result<u8, I2cError> {
        let mut buf = [0u8; 1];
        self.read_regs(address, reg, &mut buf).await?;
        Ok(buf[0])
    }
}

/// Factory trimming parameters, named as in the BME280 datasheet.
struct Bme280Calibration {
    t1: u16,
    t2: i16,
    t3: i16,
    p1: u16,
    p2: i16,
    p3: i16,
    p4: i16,
    p5: i16,
    p6: i16,
    p7: i16,
    p8: i16,
    p9: i16,
    h1: u8,
    h2: i16,
    h3: u8,
    h4: i16,
    h5: i16,
    h6: i8,
}

impl Bme280Calibration {
    fn parse(tp: &[u8; 26], h: &[u8; 7]) -> Self {
        let u = |i: usize| u16::from_le_bytes([tp[i], tp[i + 1]]);
        let s = |i: usize| i16::from_le_bytes([tp[i], tp[i + 1]]);
        Self {
            t1: u(0),
            t2: s(2),
            t3: s(4),
            p1: u(6),
            p2: s(8),
            p3: s(10),
            p4: s(12),
            p5: s(14),
            p6: s(16),
            p7: s(18),
            p8: s(20),
            p9: s(22),
            h1: tp[25],
            h2: i16::from_le_bytes([h[0], h[1]]),
            h3: h[2],
            // dig_H4 and dig_H5 are 12-bit values sharing the nibbles of 0xE5
            h4: ((h[3] as i8 as i16) * 16) | (h[4] & 0x0F) as i16,
            h5: ((h[5] as i8 as i16) * 16) | (h[4] >> 4) as i16,
            h6: h[6] as i8,
        }
    }

    /// Convert raw ADC values using the floating-point formulas from the
    /// datasheet (section 8.1).
    fn compensate(&self, raw: &[u8; 8]) -> Result<EnvReading, SensorError> {
        let adc_p = ((raw[0] as u32) << 12 | (raw[1] as u32) << 4 | (raw[2] as u32) >> 4) as f64;
        let adc_t = ((raw[3] as u32) << 12 | (raw[4] as u32) << 4 | (raw[5] as u32) >> 4) as f64;
        let adc_h = ((raw[6] as u32) << 8 | raw[7] as u32) as f64;

        let (t1, t2, t3) = (self.t1 as f64, self.t2 as f64, self.t3 as f64);
        let var1 = (adc_t / 16384.0 - t1 / 1024.0) * t2;
        let dt = adc_t / 131072.0 - t1 / 8192.0;
        let var2 = dt * dt * t3;
        let t_fine = var1 + var2;
        let temperature = t_fine / 5120.0;

        let mut var1 = t_fine / 2.0 - 64000.0;
        let mut var2 = var1 * var1 * self.p6 as f64 / 32768.0;
        var2 += var1 * self.p5 as f64 * 2.0;
        var2 = var2 / 4.0 + self.p4 as f64 * 65536.0;
        var1 = (self.p3 as f64 * var1 * var1 / 524288.0 + self.p2 as f64 * var1) / 524288.0;
        var1 = (1.0 + var1 / 32768.0) * self.p1 as f64;
        if var1 == 0.0 {
            return Err(SensorError::InvalidData);
        }
        let mut pressure = 1048576.0 - adc_p;
        pressure = (pressure - var2 / 4096.0) * 6250.0 / var1;
        let var1 = self.p9 as f64 * pressure * pressure / 2147483648.0;
        let var2 = pressure * self.p8 as f64 / 32768.0;
        pressure += (var1 + var2 + self.p7 as f64) / 16.0;

        let mut humidity = t_fine - 76800.0;
        humidity = (adc_h - (self.h4 as f64 * 64.0 + self.h5 as f64 / 16384.0 * humidity))
            * (self.h2 as f64 / 65536.0
                * (1.0
                    + self.h6 as f64 / 67108864.0
                        * humidity
                        * (1.0 + self.h3 as f64 / 67108864.0 * humidity)));
        humidity *= 1.0 - self.h1 as f64 * humidity / 524288.0;
        let humidity = humidity.clamp(0.0, 100.0);

        Ok(EnvReading {
            temperature_c: temperature as f32,
            humidity_percent: humidity as f32,
            pressure_pa: pressure as f32,
        })
    }
}

struct Bme280 {
    bus: SharedI2c,
    calibration: Bme280Calibration,
}

impl Bme280 {
    /// Check the chip ID and load the factory calibration.
    async fn init(bus: SharedI2c) -> Result<Self, SensorError> {
        if bus.read_reg(BME280_ADDR, BME280_CHIP_ID).await? != BME280_CHIP_ID_VALUE {
            return Err(SensorError::NoDevice);
        }

        let mut tp = [0u8; 26];
        bus.read_regs(BME280_ADDR, BME280_CALIB_TP, &mut tp).await?;
        let mut h = [0u8; 7];
        bus.read_regs(BME280_ADDR, BME280_CALIB_H, &mut h).await?;

        // ctrl_hum only takes effect on the next ctrl_meas write, which every read does
        bus.write(BME280_ADDR, &[BME280_CTRL_HUM, BME280_CTRL_HUM_VALUE])
            .await?;

        Ok(Self {
            bus,
            calibration: Bme280Calibration::parse(&tp, &h),
        })
    }
}

impl EnvSensor for Bme280 {
    fn read<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<EnvReading, SensorError>> + 'a>> {
        Box::pin(async move {
            self.bus
                .write(BME280_ADDR, &[BME280_CTRL_MEAS, BME280_CTRL_MEAS_FORCED])
                .await?;

            for _ in 0..BME280_MAX_POLLS {
                Timer::after_millis(BME280_POLL_INTERVAL_MS).await;
                let status = self.bus.read_reg(BME280_ADDR, BME280_STATUS).await?;
                if status & BME280_STATUS_MEASURING == 0 {
                    let mut raw = [0u8; 8];
                    self.bus
                        .read_regs(BME280_ADDR, BME280_DATA, &mut raw)
                        .await?;
                    return self.calibration.compensate(&raw);
                }
            }

            Err(SensorError::Timeout)
        })
    }
}

struct Lis3dh {
    bus: SharedI2c,
}

impl Lis3dh {
    /// Check the chip ID and start continuous sampling.
    async fn init(bus: SharedI2c) -> Result<Self, SensorError> {
        if bus.read_reg(LIS3DH_ADDR, LIS3DH_WHO_AM_I).await? != LIS3DH_WHO_AM_I_VALUE {
            return Err(SensorError::NoDevice);
        }

        bus.write(LIS3DH_ADDR, &[LIS3DH_CTRL_REG1, LIS3DH_CTRL_REG1_VALUE])
            .await?;
        bus.write(LIS3DH_ADDR, &[LIS3DH_CTRL_REG4, LIS3DH_CTRL_REG4_VALUE])
            .await?;

        Ok(Self { bus })
    }
}

impl Accelerometer for Lis3dh {
    fn read<'a>(
        &'a mut self,
    ) -> Pin<Box<dyn Future<Output = Result<Acceleration, SensorError>> + 'a>> {
        Box::pin(async move {
            let mut raw = [0u8; 6];
            self.bus
                .read_regs(
                    LIS3DH_ADDR,
                    LIS3DH_OUT_X_L | LIS3DH_AUTO_INCREMENT,
                    &mut raw,
                )
                .await?;

            // Samples are 12-bit, left-justified in little-endian 16-bit words
            let axis = |i: usize| {
                (i16::from_le_bytes([raw[i], raw[i + 1]]) >> 4) as f32 * LIS3DH_G_PER_DIGIT
            };
            Ok(Acceleration {
                x: axis(0),
                y: axis(2),
                z: axis(4),
            })
        })
    }
}

pub struct EnvSenseDriver;

impl DriverMeta for EnvSenseDriver {
    const ID: ModuleID = ModuleID {
        md0: ModuleDetectResistor::R1K,
        md1: ModuleDetectResistor::R22K,
    };
}

impl<G: BankPins> Driver<G> for EnvSenseDriver {
    async fn create(
        gpio_bank: GpioBank<G>,
        slot: ModuleSlot,
        registry: &mut Registry,
        bus_allocator: &mut BusAllocator,
    ) -> Result<(), DriverError> {
        // Both sensors sit on GPIO0 (SCL) and GPIO1 (SDA)
        let bus = bus_allocator
            .create_i2c_bitbang(
                gpio_bank.gpio0.into(),
                gpio_bank.gpio1.into(),
                I2C_FREQUENCY_HZ,
            )
            .map_err(|_| DriverError::InitFailed)?;
        let bus = Arc::new(Mutex::new(bus));

        let bme280 = Bme280::init(SharedI2c(bus.clone()))
            .await
            .map_err(|_| DriverError::InitFailed)?;
        let lis3dh = Lis3dh::init(SharedI2c(bus))
            .await
            .map_err(|_| DriverError::InitFailed)?;

        // Only register once both sensors answered, so apps never see a half-initialised module
        registry.register(
            slot,
            Self::ID,
            Box::new(bme280) as Box<dyn EnvSensor>,
        );
        registry.register(
            slot,
            Self::ID,
            Box::new(lis3dh) as Box<dyn Accelerometer>,
        );

        // D1 is driven active-high from GPIO9 through R3
        registry.register(
            slot,
            Self::ID,
            pin_led::<Generic>(gpio_bank.gpio9.into(), false),
        );

        Ok(())
    }
}