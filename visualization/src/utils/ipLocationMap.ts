/**
 * 将笔记 ip_location 与 ECharts 内置 china.json（省/直辖市/自治区简称）对齐。
 * 参考 echarts@4.9.0 map/json：properties.name 多为「广东」「北京」等形式。
 */
const FULLNAME_TO_MAPNAME: Record<string, string> = {
  北京市: '北京',
  上海市: '上海',
  天津市: '天津',
  重庆市: '重庆',
  广东省: '广东',
  浙江省: '浙江',
  江苏省: '江苏',
  山东省: '山东',
  河南省: '河南',
  四川省: '四川',
  湖北省: '湖北',
  湖南省: '湖南',
  福建省: '福建',
  安徽省: '安徽',
  河北省: '河北',
  陕西省: '陕西',
  辽宁省: '辽宁',
  江西省: '江西',
  云南省: '云南',
  广西壮族自治区: '广西',
  广西省: '广西',
  山西省: '山西',
  吉林省: '吉林',
  黑龙江省: '黑龙江',
  贵州省: '贵州',
  甘肃省: '甘肃',
  内蒙古自治区: '内蒙古',
  新疆维吾尔自治区: '新疆',
  西藏自治区: '西藏',
  宁夏回族自治区: '宁夏',
  海南省: '海南',
  青海省: '青海',
  香港特别行政区: '香港',
  澳门特别行政区: '澳门',
  台湾省: '台湾',
}

/** 常见城市显示名归并到省级地图块 */
const CITY_TO_PROVINCE: Record<string, string> = {
  广州: '广东',
  深圳: '广东',
  杭州: '浙江',
  南京: '江苏',
  苏州: '江苏',
  成都: '四川',
  武汉: '湖北',
  西安: '陕西',
  郑州: '河南',
  长沙: '湖南',
}

export function toMapRegionName(raw: string | undefined | null): string | null {
  if (!raw || !raw.trim()) return null
  const s = raw.trim()
  if (s === '未知IP') return null
  if (FULLNAME_TO_MAPNAME[s]) return FULLNAME_TO_MAPNAME[s]
  if (CITY_TO_PROVINCE[s]) return CITY_TO_PROVINCE[s]
  return s
}

export function sameIpRegion(a: string | null, b: string | undefined | null): boolean {
  if (!a || !b) return false
  if (a === b) return true
  const na = toMapRegionName(a)
  const nb = toMapRegionName(b)
  if (na && nb) return na === nb
  return false
}
