/**
 * 高德地图天气封装，可以返回带图标url的天气对象👍
 * 您可能需要修改的地方：
 * 1, 地图KEY, VERSION
 * 2, 变量 iconWeatherMap
 * 3, function getIcon 中的url路径
 * @author tjn
 * @date 2021-09-10
 * @see https://pan.baidu.com/s/14Qo0uS7a9Oz5ZhHgw6TpyA 提取码: 4qjh  所有图标
 */

// 安装：npm i @amap/amap-jsapi-loader --save
import AMapLoader from '@amap/amap-jsapi-loader'
import { get } from 'http'

// 地图key和版本
const KEY = 'your amap key'
const VERSION = '2.0'

// 高德地图对象
let AMap = null
// 天气实例对象
let weather = null

/**
 * 图标和天气现象的映射关系
 * key: 某一类天气现象，这个key会和icon文件名一一对应
 * value: 某一类下天气现象，可以按照自己的需求划分分类
 * @see https://lbs.amap.com/api/webservice/guide/tools/weather-code
 */
const iconWeatherMap = {
  '风': ['有风', '平静', '微风', '和风', '清风', '强风/劲风', '疾风', '大风', '烈风', '风暴', '狂爆风', '飓风', '热带风暴', '龙卷风'],
  '多云': ['少云', '晴间多云', '多云'],
  '雪': ['雪', '阵雪', '小雪', '中雪', '大雪', '暴雪', '小雪-中雪', '中雪-大雪', '大雪-暴雪', '冷'],
  '雾': ['浮尘', '扬沙', '沙尘暴', '强沙尘暴', '雾', '浓雾', '强浓雾', '轻雾', '大雾', '特强浓雾'],
  '晴': ['晴', '热'],
  '雨夹雪': ['雨雪天气', '雨夹雪', '阵雨夹雪'],
  '雨': ['阵雨', '雷阵雨', '雷阵雨并伴有冰雹', '小雨', '中雨', '大雨', '暴雨', '大暴雨', '特大暴雨', '强阵雨', '强雷阵雨', '极端降雨', '毛毛雨/细雨', '雨', '小雨-中雨', '中雨-大雨', '大雨-暴雨', '暴雨-大暴雨', '大暴雨-特大暴雨', '冻雨'],
  '阴': ['阴', '霾', '中度霾', '重度霾', '严重霾', '未知']
}

/**
 * 加载地图并引入天气组件，这里是第一次加载地图，需要Key，version
 * 如果在后面的业务中，这里不是第一次加载地图，key和version可以省略
 * @see https://lbs.amap.com/api/jsapi-v2/guide/abc/load
 */
function _initWeather () {
  console.debug('---init weather---')

  return AMapLoader.load({
    key: KEY,
    version: VERSION,
    plugins: ['AMap.Weather']
  }).then((Map) => {
    // 保存地图对象
    AMap = Map
  }).catch(e => {
    console.error('初始化地图/天气失败 = ', e)
  })
}

/**
 * 根据天气现象返回其图标icon url
 * @param {String} weather 天气现象
 * @returns 天气现象对应的某一类的url
 */
function _getIcon (weather) {
  // 这个是icon的默认值
  let url = require('@/assets/images/icon/weather/阴.png')

  for (const weatherKey in iconWeatherMap) {
    if (Object.hasOwnProperty.call(iconWeatherMap, weatherKey)) {
      const weatherNames = iconWeatherMap[weatherKey]
      const findWeatherItem = weatherNames.find(name => weather === name)

      // 如果找了某一类的图标了，那重新赋值url
      if (findWeatherItem) {
        // 这里的weatherKey和icon的名字一一对应了
        url = require(`@/assets/images/icon/weather/${weatherKey}.png`)
        // console.debug('@find weather key = ', weatherKey)
        break
      }
    }
  }

  return url
}

/**
 * 查询目标城市/区域的天气预报状况。
 * @param {Number | String} adcode 城市名称、区域编码（如『上海市』、『310000』），默认上海
 * @see https://lbs.amap.com/api/jsapi-v2/guide/services/weather
 * @see https://lbs.amap.com/api/webservice/download
 * @returns 返回一个Promise
 */
export async function getWeather (adcode = 310000) {
  // 如果没有实例的话那么初始化一下
  if (!AMap) {
    await _initWeather()
  }

  return new Promise((resolve, reject) => {
    if (!weather) {
      weather = new AMap.Weather()
    }
    weather.getLive(adcode, (err, data) => {
      if (!err) {
        // 组装新的天气结果对象，除了高德地图天气自带的属性，这里面还包含了图标的地址属性：url
        const weatherData = Object.assign({}, data, { url: _getIcon(data.weather) })
        resolve(weatherData)
      } else {
        console.error('获取天气失败 = ', err)
        reject(err)
      }
    })
  })
}

getWeather().then(data => {
    console.log('---weather data---', data)
    console.log(`@weather = ${data}, @icon = ${data.url}`)
})
