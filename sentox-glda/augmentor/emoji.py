from pathlib import Path
from tqdm import tqdm
import warnings
import time
import re
import jieba
import jieba.posseg as pseg


class EmojiAugmentor:
    """Comprehensive emoji augmentation with extensive Chinese evasion patterns"""

    def __init__(self, intensity: str = 'light'):
        self.intensity = intensity

        # Chinese character-emoji mapping dictionary
        self.char_emoji_map = {

            '我': ['🐧', '👤', '🙋‍♂️', '🙋‍♀️', 'wǒ', 'wo', 'w', '👨‍💻', '🤖', '俺', '本人', '在下', '小的', '鄙人', '老子',
                   '姐', '爷', '哥', '姐姐', '妹妹', '大爷', '老娘', 'me', 'I', '咱', '偶', '窝', '伦家', '人家', '鹅',
                   '额', '俺们', '咋'],
            '你': ['🫵', '👉', 'nǐ', 'ni', 'u', 'U', '您', '你丫', '尼', '泥', '妮', '拟', '逆', '倪', '你个', '你这个',
                   '你咋', 'you', '汝', '君', '阁下', '老兄', '兄弟', '姐妹', '亲', '宝贝', '同学', '朋友', '尼玛',
                   '泥萌', '银', '银家', '粑粑', '麻麻'],
            '他': ['👨', '🧍‍♂️', 'tā', 'ta', 'T', '她', '它', 'TA', '伊', '其', '那人', '此君', '那位', '这位', '那个人',
                   '这个人', 'he', 'him', '某人', '此人', '那货', '这货', '那家伙', '这家伙'],
            '她': ['👩', '🧍‍♀️', 'tā', 'ta', 'T', '他', 'TA', '伊', '其', '那女', '此女'],
            '我们': ['👥', '👫', '👬', '👭', 'wǒmen', 'we', '咱们', '咱', '俺们'],
            '你们': ['👥', '🫵', 'nǐmen', 'you', '你丫们', '各位', '大家'],
            '他们': ['👥', '👨‍👨‍👦', 'tāmen', 'they', '她们', '它们', '这些人'],

            '爱': ['❤️', '💕', '💖', '♥️', '💗', '💝', 'ài', 'ai', 'love', '❣️', '💘', '💟', '😍', '🥰', '😘', '💋', '🤟', '💞', '💓', '爱死了', '超爱', '巨爱', '贼爱', '老爱了', 'luv', '耐', '爱你', '爱死', '爱惨了'],
            '恨': ['😡', '😠', '💢', '👿', 'hèn', 'hen', 'hate', '🤬', '😤', '💯', '⚡', '🔥', '恨死', '讨厌死', '气死', '烦死', '讨厌', 'fxxk', '草', '日', '干', '恨不得', '恨死了', '气炸', '炸了'],
            '喜欢': ['😍', '🥰', '😘', '💕', 'xǐhuān', 'like', '❤️', '💖', '👍', '😊', '爱', '中意', '钟意', '稀饭', '西欢', '系欢', '喜欢死了', '爱惨了', '爱死了', '喜欢到爆', '喜欢得不行', '爱到不行'],
            '讨厌': ['😒', '🙄', '😤', 'tǎoyàn', 'hate', '💢', '😡', '🤮', '👎', '烦', '恶心', '呕', '吐了', '讨厌死了', '烦死了', '恶心死了', '受不了', '无语了', '醉了', '服了'],
            '开心': ['😊', '😁', '😄', '🥳', '😆', 'kāixīn', 'happy', '🎉', '😃', '😀', '🤗', '爽', '嗨', '开森', '开心死了', '爽歪歪', '美滋滋', '乐死了', '笑死', '哈皮', 'happy到飞起'],
            '难过': ['😢', '😭', '💧', 'nánguò', 'sad', '😞', '☔', '😟', '😔', '💔', '心碎', '桑心', '伤心', '难受', '痛苦', '郁闷', '蓝瘦', '想哭', '泪目', '心疼', '心痛', '痛哭', '哭唧唧'],
            '生气': ['😡', '😠', '💢', '👿', 'shēngqì', 'angry', '🤬', '😤', '💥', '⚡', '气死', '炸了', '怒', '火大', '暴躁', '气炸', '气到爆', '怒了', '发火', '抓狂', '疯了'],
            '害怕': ['😨', '😰', '😱', 'hàipà', 'fear', '👻', '🙀', '😵', '😓', '怕怕', '吓死', '恐怖', '瑟瑟发抖', '怂', '虚', '慌得一批', '吓尿', '吓傻', '惊恐', '恐惧'],
            '惊讶': ['😲', '😯', '🤯', 'jīngyà', 'surprise', '😮', '🤭', '😳'],
            '高兴': ['😄', '😁', '🥳', 'gāoxìng', 'joy', '🎊', '🎉', '😊', '🤩'],

            '说': ['💬', '🗣️', '📢', 'shuō', 'say', '🔊', '📣', '🎤', '👄', '🗨️', '💭'],
            '看': ['👀', '👁️', '🔍', 'kàn', 'see', '👓', '🕵️‍♂️', '🧐', '👁️‍🗨️', '🔭'],
            '听': ['👂', '🎧', '🔊', 'tīng', 'listen', '👂🏻', '🎵', '🎶', '📻', '🔉'],
            '想': ['🤔', '💭', '🧠', 'xiǎng', 'think', '💡', '🎯', '🤯', '💫'],
            '做': ['🔨', '⚙️', '🛠️', 'zuò', 'do', '👷‍♂️', '🏗️', '⚒️', '🔧', '🪛'],
            '吃': ['🍽️', '🥢', '👄', 'chī', 'eat', '🍚', '🥘', '🍜', '🍴', '🥄'],
            '喝': ['🍺', '🥤', '☕', 'hē', 'drink', '🧊', '🥛', '🍵', '🥃', '🍷'],
            '走': ['🚶‍♂️', '👣', '🏃‍♂️', 'zǒu', 'walk', '🚶‍♀️', '👟', '🦶', '🥾'],
            '跑': ['🏃‍♂️', '💨', '⚡', 'pǎo', 'run', '🏃‍♀️', '🏃', '👟', '🏃‍♂️'],
            '来': ['➡️', '👋', '🚶‍♂️', 'lái', 'come', '🏃‍♂️', '📍', '🎯'],
            '去': ['⬅️', '🚶‍♂️', '🏃‍♂️', 'qù', 'go', '➡️', '🎯', '📍'],
            '买': ['🛒', '💰', '🛍️', 'mǎi', 'buy', '💳', '🏪', '💸'],
            '卖': ['🏪', '💰', '📊', 'mài', 'sell', '💸', '🛒', '💳'],
            '学': ['📚', '🎓', '✏️', 'xué', 'study', '📖', '🧑‍🎓', '✍️'],
            '教': ['👩‍🏫', '📚', '✏️', 'jiào', 'teach', '🎓', '📖', '👨‍🏫'],
            '工作': ['💼', '🏢', '👔', 'gōngzuò', 'work', '⚙️', '🛠️', '💻'],
            '休息': ['😴', '🛏️', '☕', 'xiūxi', 'rest', '🛋️', '😌', '🧘‍♂️'],

            '钱': ['💰', '💸', '💵', '💴', '💶', '💷', 'qián', 'money', '💳', '🏦', '🪙', '💎'],
            '车': ['🚗', '🚙', '🚘', 'chē', 'car', '🚕', '🚐', '🏎️', '🚓', '🚑'],
            '房': ['🏠', '🏡', '🏢', 'fáng', 'house', '🏘️', '🏰', '🏛️', '🏬', '🏭'],
            '书': ['📚', '📖', '📕', 'shū', 'book', '📗', '📘', '📙', '📓', '📔'],
            '电脑': ['💻', '🖥️', 'diànnǎo', 'computer', '⌨️', '🖱️', '💾', '💿'],
            '手机': ['📱', '☎️', 'shǒujī', 'phone', '📞', '📟', '📲', '☎️'],
            '电视': ['📺', '📻', 'diànshì', 'tv', '🖥️', '📽️', '🎬'],
            '音乐': ['🎵', '🎶', '🎤', 'yīnyuè', 'music', '🎧', '🎼', '🎹'],
            '电影': ['🎬', '🎭', '📽️', 'diànyǐng', 'movie', '🎥', '🍿', '🎪'],
            '游戏': ['🎮', '🕹️', '🎯', 'yóuxì', 'game', '🎲', '🃏', '🎰'],
            '衣服': ['👕', '👚', '👔', 'yīfu', 'clothes', '👗', '👖', '🧥'],
            '鞋子': ['👟', '👞', '👠', 'xiézi', 'shoes', '🥾', '👡', '🩴'],
            '食物': ['🍎', '🍞', '🍚', 'shíwù', 'food', '🥘', '🍜', '🥗'],
            '水': ['💧', '🌊', '🚰', 'shuǐ', 'water', '💦', '🏊‍♂️', '⛲'],

            '好': ['👍', '👌', '✅', 'hǎo', 'good', '💯', '🔥', '👏', '🎉', '😊', '👏', '🥇'],
            '坏': ['👎', '❌', '💩', 'huài', 'bad', '🚫', '⛔', '🔴', '😞', '👎'],
            '大': ['⬆️', '📏', '🔥', 'dà', 'big', '📈', '💪', '🦣', '🐘', '📐'],
            '小': ['⬇️', '🤏', '👶', 'xiǎo', 'small', '🐭', '🤱', '🐁', '📉'],
            '美': ['😍', '🌸', '💖', 'měi', 'beautiful', '👸', '🦋', '✨', '💅', '🌺'],
            '丑': ['🤮', '👹', '💩', 'chǒu', 'ugly', '👺', '😵', '🤡', '👻'],
            '快': ['⚡', '💨', '🏃‍♂️', 'kuài', 'fast', '⏱️', '🚀', '🏎️', '💫'],
            '慢': ['🐌', '⏳', '😴', 'màn', 'slow', '🦥', '⌛', '🚶‍♂️', '🐢'],
            '高': ['⬆️', '🗼', '🏔️', 'gāo', 'high', '🏗️', '🚁', '🪂', '🎢'],
            '低': ['⬇️', '📉', '🔽', 'dī', 'low', '🚇', '🕳️', '⬇️'],
            '新': ['🆕', '✨', '🎁', 'xīn', 'new', '👶', '🌱', '🔆'],
            '老': ['👴', '👵', '⏰', 'lǎo', 'old', '🕰️', '📜', '🏚️'],
            '热': ['🔥', '🌡️', '☀️', 'rè', 'hot', '🥵', '♨️', '🌶️'],
            '冷': ['❄️', '🥶', '🧊', 'lěng', 'cold', '🌨️', '⛄', '🥤'],
            '亮': ['💡', '✨', '🌟', 'liàng', 'bright', '🔆', '💫', '⭐'],
            '暗': ['🌑', '🕳️', '🔦', 'àn', 'dark', '🌚', '🖤', '⚫'],

            '牛': ['🐄', '🐂', '💪', 'niú', 'cow', 'bull', '🤘', '👑', '🔥'],
            '屌': ['🍆', '🔥', '💪', 'diǎo', '🚀', '👑', '⚡', '💥'],
            '草': ['🌱', '🌿', '🍃', 'cǎo', 'grass', '🌾', '☘️', '🌳'],
            '操': ['🤬', '💢', '😡', 'cāo', '🔥', '💥', '⚡', '🌶️'],
            '傻': ['🤪', '🙃', '😵', 'shǎ', 'silly', '🤡', '😜', '🧠'],
            '笨': ['🤪', '🙃', '😵', 'bèn', 'stupid', '🤡', '😜', '🧠'],
            '蠢': ['🤪', '🙃', '😵', 'chǔn', 'dumb', '🤡', '😜', '🧠'],

            '今天': ['📅', '☀️', 'jīntiān', 'today', '🌅', '🌞', '🗓️'],
            '昨天': ['📆', '🌙', 'zuótiān', 'yesterday', '🌜', '⏮️'],
            '明天': ['📅', '🌅', 'míngtiān', 'tomorrow', '🌄', '⏭️'],
            '现在': ['⏰', '🕐', 'xiànzài', 'now', '⏱️', '🕰️', '📍'],
            '以前': ['📜', '⏪', 'yǐqián', 'before', '🕰️', '⌛'],
            '以后': ['⏩', '🔮', 'yǐhòu', 'later', '🚀', '⏭️'],
            '早上': ['🌅', '☀️', 'zǎoshang', 'morning', '🐓', '☕'],
            '晚上': ['🌙', '🌃', 'wǎnshang', 'evening', '💤', '🌆'],
            '中午': ['☀️', '🍽️', 'zhōngwǔ', 'noon', '🕐', '☀️'],

            '头': ['🗣️', '👤', 'tóu', 'head', '🧠', '💇‍♂️', '👨‍🦲'],
            '脸': ['😀', '👤', 'liǎn', 'face', '😊', '😢', '🤡'],
            '眼睛': ['👀', '👁️', 'yǎnjīng', 'eyes', '👓', '🕶️', '👁️‍🗨️'],
            '鼻子': ['👃', '🐽', 'bízi', 'nose', '👃🏻', '🤧'],
            '嘴': ['👄', '💋', 'zuǐ', 'mouth', '😮', '🤐', '🗣️'],
            '手': ['🙌', '👏', 'shǒu', 'hand', '✋', '👋', '🤝'],
            '脚': ['🦶', '👣', 'jiǎo', 'foot', '👟', '🩴', '🦵'],

            '爸爸': ['👨‍👧‍👦', '👔', 'bàba', 'dad', '👨', '🧔', '👪'],
            '妈妈': ['👩‍👧‍👦', '👸', 'māma', 'mom', '👩', '🤱', '👪'],
            '儿子': ['👦', '🧒', 'érzi', 'son', '👨‍👦', '🎮'],
            '女儿': ['👧', '🧒', 'nǚér', 'daughter', '👩‍👧', '🎀'],
            '哥哥': ['👨', '🧔', 'gēge', 'brother', '👑', '💪'],
            '姐姐': ['👩', '👸', 'jiějie', 'sister', '💄', '💅'],
            '弟弟': ['👦', '🧒', 'dìdi', 'brother', '🎮', '👨‍💻'],
            '妹妹': ['👧', '🧏‍♀️', 'mèimei', 'sister', '🎀', '👱‍♀️'],

            '狗': ['🐶', '🐕', 'gǒu', 'dog', '🦮', '🐕‍🦺', '🐩'],
            '猫': ['🐱', '🐈', 'māo', 'cat', '🐾', '😺', '🙀'],
            '鸟': ['🐦', '🕊️', 'niǎo', 'bird', '🦅', '🐧', '🦜'],
            '鱼': ['🐟', '🐠', 'yú', 'fish', '🎣', '🐡', '🦈'],
            '猪': ['🐷', '🐽', 'zhū', 'pig', '🥓', '🐖'],
            '羊': ['🐑', '🐏', 'yáng', 'sheep', '🐐', '🧶'],

            '红': ['🔴', '❤️', 'hóng', 'red', '🌹', '🍎', '🚗'],
            '绿': ['🟢', '🌿', 'lǜ', 'green', '🌱', '🥬', '🔋'],
            '蓝': ['🔵', '💙', 'lán', 'blue', '🌊', '💎', '🧿'],
            '黄': ['🟡', '💛', 'huáng', 'yellow', '🌞', '🍌', '⚡'],
            '黑': ['⚫', '🖤', 'hēi', 'black', '🌚', '🖤', '⬛'],
            '白': ['⚪', '🤍', 'bái', 'white', '☁️', '❄️', '⬜'],

            '一': ['1️⃣', '①', '壹', 'yī', 'one', 'I', '🥇'],
            '二': ['2️⃣', '②', '贰', 'èr', 'two', 'II', '🥈'],
            '三': ['3️⃣', '③', '叁', 'sān', 'three', 'III', '🥉'],
            '四': ['4️⃣', '④', '肆', 'sì', 'four', 'IV', '🍀'],
            '五': ['5️⃣', '⑤', '伍', 'wǔ', 'five', 'V', '✋'],
            '六': ['6️⃣', '⑥', '陆', 'liù', 'six', 'VI', '🎲'],
            '七': ['7️⃣', '⑦', '柒', 'qī', 'seven', 'VII', '🌈'],
            '八': ['8️⃣', '⑧', '捌', 'bā', 'eight', 'VIII', '♾️'],
            '九': ['9️⃣', '⑨', '玖', 'jiǔ', 'nine', 'IX', '🌟'],
            '十': ['🔟', '⑩', '拾', 'shí', 'ten', 'X', '💯'],

            '政府': ['🏛️', 'zhèngfǔ', 'gov', '🏢', '⚖️', 'ZF', 'government'],
            '官员': ['👔', '🎩', 'guānyuán', '👨‍💼', '💼', 'GY', 'official'],
            '领导': ['👑', '🎭', 'lǐngdǎo', '👨‍💼', '💺', 'LD', 'leader'],
            '警察': ['👮‍♂️', '🚔', 'jǐngchá', '👮‍♀️', '🚨', 'JC', 'police'],
            '抗议': ['✊', '📢', 'kàngyì', '🗣️', '📣', 'KY', 'protest'],
            '批评': ['💬', '🗯️', 'pīpíng', '📝', '✍️', 'PP', 'criticize'],
            '示威': ['✊', '📢', 'shìwēi', '🚩', '🪧', 'SW', 'demonstrate'],
            '民主': ['🗳️', '⚖️', 'mínzhǔ', '🏛️', '📊', 'MZ', 'democracy'],
            '自由': ['🕊️', '🦅', 'zìyóu', '🌟', '⭐', 'ZY', 'freedom'],

            '牛逼': ['🐄', '💪', '🔥', '👑', 'niúbī', 'nb', 'NB', '6666', '厉害', '绝了', '牛批', '牛掰', '牛B', '牛比', '厉害了', '太强了', '神了', '无敌', 'awesome', 'amazing', '吊炸天', '炸裂', '燃爆'],
            '666': ['👑', '🔥', '💪', '🎯', '🏆', '⚡', '💯', '牛逼', '厉害', '溜', '6翻了', '666666', '溜得飞起', '厉害了我的哥', '强无敌', 'nice', 'good', '可以的', '溜溜溜', '牛牛牛'],
             '哈哈': ['😂', '🤣', '😄', 'hāhā', 'lol', '233', '笑', '2333', '233333', 'hhhh', 'hhhhh', 'lmao', '哈哈哈', '笑死我了', '笑疯了', '笑岔气', '笑不活了', '笑飞了', '哈哈哈哈', '嘿嘿', '嘻嘻'],
            '绝了': ['🔥', '💯', '👑', 'juéle', '牛逼', '厉害', '666', '太强了', '绝绝子', '绝美', '绝杀', '绝赞', '绝妙', '绝佳', '绝世', '绝代', '绝顶', '绝伦', '绝无仅有', '绝对', '太绝了'],
            '裂开': ['💥', '😵', '🤯', 'lièkāi', '崩溃', '震惊', '炸了', '裂了', '我裂开了', '直接裂开', '当场裂开', '瞬间裂开', '心态裂开', '人都裂开了', '整个人裂开', '笑裂开了'],
            '炸了': ['💥', '🔥', '🤯', 'zhàle', '火了', '爆了', '裂开', '燃爆', '爆炸', '炸裂', '炸天', '炸翻', '直接炸了', '当场炸了', '瞬间炸了', '笑炸了', '气炸了'],
            '寄了': ['📦', '💀', '😵', 'jìle', '完了', '凉了', '死了', '没了', '废了', 'over了', 'gg了', '结束了', '拜拜了', '再见了', '告辞了'],
            '摆了': ['🤷‍♂️', '😑', '放弃', 'bǎile', '算了', '不管了', '躺了', '随便了', '无所谓了', '懒得管了', '爱咋咋地', '摆烂了', '开摆了'],
            '躺平': ['😴', '🛏️', '🤷‍♂️', 'tǎngpíng', '摆烂', '放弃', '佛系', '躺了', '平躺', '直接躺平', '彻底躺平', '选择躺平', '开始躺平', '躺得很平'],

            '233': ['😂', '🤣', '😄', '哈哈', 'lol', '笑死', '笑cry'],
            '2333': ['😂', '🤣', '😄', '233', 'lmao', '哈哈哈'],
            '蛤蛤': ['😂', '🤣', '🐸', 'hāhā', '哈哈', '呵呵'],
            '呵呵': ['😏', '🙄', '😒', 'hēhē', '冷笑', '无语'],
            '嘿嘿': ['😏', '😈', '🤭', 'hēihēi', '坏笑', '嘿嘿嘿'],
            '嘻嘻': ['😊', '😁', '🤭', 'xīxī', '偷笑', '开心'],
            '哼哼': ['😤', '🙄', '😒', 'hēnghēng', '不满', '生气'],

            '火了': ['🔥', '💥', '🌟', 'huǒle', '红了', '爆红', '炸了'],
            '凉了': ['❄️', '😵', '💀', 'liángle', '完了', '死了', '寄了'],
            '摆烂': ['🤷‍♂️', '😑', '💩', 'bǎilàn', '躺平', '放弃', '烂泥'],
            '内卷': ['🌀', '😵‍💫', '🔄', 'nèijuǎn', '竞争', '焦虑', '卷'],
            '打工人': ['👷‍♂️', '💼', '😭', 'dǎgōngrén', '上班族', '社畜', '工人'],
            '社畜': ['🐄', '💼', '😭', 'shèchù', '打工人', '上班族', '牛马'],
            '牛马': ['🐄', '🐎', '😭', 'niúmǎ', '社畜', '打工人', '苦逼'],

            'GG': ['💀', '😵', '结束', 'game over', '完了', '死了'],
            'AFK': ['🚶‍♂️', '⏰', 'away', '离开', '暂离'],
            'BUG': ['🐛', '❌', '错误', 'bug', '故障'],
            'FPS': ['🎮', '🔫', '射击', 'fps', '帧数'],
            'RPG': ['🗡️', '🛡️', '角色', 'rpg', '游戏'],
            '开黑': ['🎮', '👥', '组队', 'kāihēi', '一起玩'],
            '菜鸟': ['🐣', '🆕', '新手', 'càiniǎo', 'noob', '萌新'],
            '大佬': ['👑', '💪', '🔥', 'dàlǎo', 'boss', '高手', '大神'],
            '大神': ['👑', '🔥', '⚡', 'dàshén', '大佬', '高手', '神'],
            '菜鸡': ['🐔', '😅', '菜', 'càijī', '菜鸟', '新手', '弱'],
            '秀': ['✨', '🔥', '💫', 'xiù', 'show', '厉害', '牛逼'],
            '翻车': ['🚗', '💥', '😵', 'fānchē', '失败', '完蛋', '翻了'],

            '点赞': ['👍', '❤️', '赞', 'diǎnzàn', 'like', '支持'],
            '关注': ['👀', '➕', '关心', 'guānzhù', 'follow', '订阅'],
            '转发': ['🔄', '📤', '分享', 'zhuǎnfā', 'retweet', 'share'],
            '私信': ['💌', '📩', 'DM', 'sīxìn', 'private message'],
            '刷屏': ['📱', '🔄', '霸屏', 'shuāpíng', 'spam', '刷'],
            '热搜': ['🔥', '📈', '🔍', 'rèsōu', 'trending', '热门'],
            '吃瓜': ['🍉', '👀', '看戏', 'chīguā', '围观', '看热闹'],
            '瓜友': ['🍉', '👥', '网友', 'guāyǒu', '吃瓜群众'],
            '反转': ['🔄', '😲', '逆转', 'fǎnzhuǎn', 'plot twist'],
            '塌房': ['🏠', '💥', '😱', 'tāfáng', '偶像崩塌', '人设崩塌'],

            '单身狗': ['🐕', '💔', '单身', 'dānshēngǒu', 'single', '光棍'],
            '脱单': ['💑', '❤️', '恋爱', 'tuōdān', '有对象了'],
            '撒狗粮': ['🐕', '💕', '秀恩爱', 'sǎgǒuliáng', 'PDA', '虐狗'],
            '柠檬精': ['🍋', '😤', '酸', 'níngméng jīng', '嫉妒', '羡慕'],
            '酸了': ['🍋', '😤', '嫉妒', 'suānle', '羡慕', '柠檬'],
            '羡慕': ['😍', '🤤', '想要', 'xiànmù', 'envy', '酸了'],
            '暖男': ['🔥', '💝', '贴心', 'nuǎnnán', 'warm guy', '好男人'],
            '直男': ['🧍‍♂️', '😐', '钢铁', 'zhínán', 'straight man'],
            '钢铁直男': ['🤖', '😐', '不解风情', 'gāngtiě zhínán', '直男'],

            '干饭': ['🍚', '🥢', '吃饭', 'gānfàn', 'eating', '恰饭'],
            '恰饭': ['🍽️', '💰', '赚钱', 'qiàfàn', '干饭', '工作'],
            '肥宅': ['🍕', '🎮', '宅男', 'féizhái', 'otaku', '宅'],
            '宅': ['🏠', '🎮', '不出门', 'zhái', 'stay home', 'otaku'],
            '佛系': ['🧘‍♂️', '😌', '随缘', 'fóxì', 'chill', '淡定'],
            '养生': ['🍵', '🧘‍♂️', '健康', 'yǎngshēng', 'wellness', '保健'],
            '熬夜': ['🌙', '😴', '晚睡', 'áoyè', 'stay up late', '夜猫子'],
            '夜猫子': ['🌙', '🐱', '熬夜', 'yèmāozi', 'night owl'],

            '颜值': ['😍', '💅', '颜', 'yánzhí', 'looks', '外貌', '脸'],
            '颜狗': ['😍', '🐕', '看脸', 'yángǒu', '外貌协会'],
            '神颜': ['😍', '👼', '美女', 'shényán', '绝世美颜', '天仙'],
            '盛世美颜': ['👸', '✨', '🌟', 'shèngshì měiyán', '超美', '神颜'],
            '素颜': ['😊', '🧴', '无妆', 'sùyán', 'no makeup', '天然'],
            '整容脸': ['🔪', '😷', '人工', 'zhěngróng liǎn', 'plastic face'],
            '锥子脸': ['📐', '😷', '瓜子脸', 'zhuīzi liǎn', 'V-shaped face'],

            '土豪': ['💰', '👑', '有钱人', 'tǔháo', 'rich', '大款'],
            '大款': ['💰', '💼', '土豪', 'dàkuǎn', 'rich guy', '有钱人'],
            '暴富': ['💰', '📈', '发财', 'bàofù', 'get rich', '一夜暴富'],
            '财神': ['💰', '🙏', '发财', 'cáishén', 'god of wealth', '好运'],
            '锦鲤': ['🐟', '🍀', '好运', 'jǐnlǐ', 'lucky', '幸运'],
            '欧皇': ['👑', '🍀', '好运', 'ōuhuáng', 'lucky person', '运气好'],
            '非酋': ['💀', '😭', '倒霉', 'fēiqiú', 'unlucky', '运气差'],
            '咸鱼': ['🐟', '😴', '废物', 'xiányú', 'useless', '躺平'],

            '学霸': ['📚', '🤓', '学神', 'xuébà', 'top student', '学神'],
            '学渣': ['📚', '😅', '差生', 'xuézhā', 'poor student', '学弱'],
            '学神': ['📚', '👑', '学霸', 'xuéshén', 'academic god', '天才'],
            '考神': ['📝', '🙏', '考试', 'kǎoshén', 'test god', '考试高手'],
            '挂科': ['❌', '😭', '不及格', 'guàkē', 'fail exam', '失败'],
            '秃头': ['👨‍🦲', '📚', '学习', 'tūtóu', 'bald', '用脑过度'],
            '肝': ['💻', '😵', '熬夜', 'gān', '拼命工作', '加班'],
            '996': ['💼', '😭', '加班', '工作制', 'work schedule'],
            '007': ['💼', '😱', '全天候', '工作制', 'extreme work'],

            '爱豆': ['⭐', '❤️', '偶像', 'àidòu', 'idol', '明星'],
            '偶像': ['⭐', '😍', '爱豆', 'ǒuxiàng', 'idol', '明星'],
            '爱豆营业': ['⭐', '💼', '工作', 'àidòu yíngyè', 'idol working'],
            '颜粉': ['😍', '👑', '粉丝', 'yánfěn', 'looks fan', '外貌粉'],
            '唯粉': ['👑', '💎', '死忠粉', 'wéifěn', 'solo fan', '只粉'],
            '黑粉': ['😡', '👿', '黑子', 'hēifěn', 'anti-fan', 'hater'],
            '脱粉': ['👋', '💔', '不粉了', 'tuōfěn', 'unstan', '取关'],
            '路人': ['🚶‍♂️', '😐', '普通人', 'lùrén', 'passerby', '路人甲'],
            '吃瓜群众': ['🍉', '👥', '围观', 'chīguā qúnzhòng', 'onlookers'],
            'YYDS': ['👑', '🔥', '💎', '永远的神', 'eternal god', '最强', '永远滴神', '永神', '真神', 'god forever'],
            '集美': ['👭', '💅', 'jímēi', 'sisters', '姐妹', 'bestie', '小姐妹', '好姐妹', '闺蜜'],
            '绝绝子': ['🔥', '💯', '👑', 'juéjuézi', '太棒了', 'amazing', '绝了', '太绝了', '绝到爆'],
            '柠檬': ['🍋', '😤', '💛', '嫉妒', '酸', '柠檬精', '我柠檬了', '酸了', '柠檬树下你和我', '我酸了'],
            '破大防': ['🛡️', '💥', '😭', 'pò dàfáng', '情绪大崩溃', '心态爆炸', '直接破大防', '瞬间破大防'],
            '潮流': ['🌊', '👗', '时尚', 'cháoliú', 'trend', 'fashion'],
            '时髦': ['💅', '✨', '潮', 'shímáo', 'fashionable', '时尚'],
            '复古': ['⏰', '📻', '怀旧', 'fùgǔ', 'retro', 'vintage'],
            '洋气': ['✈️', '🌍', '时髦', 'yángqì', 'western style', '潮'],
            '土': ['🌍', '😅', '不时尚', 'tǔ', 'uncool', '老土'],
            '老土': ['👴', '📻', '过时', 'lǎotǔ', 'outdated', '土'],

            '网红': ['📱', '⭐', '红人', 'wǎnghóng', 'internet celebrity'],
            '主播': ['🎤', '📺', '直播', 'zhǔbō', 'streamer', 'broadcaster'],
            '直播': ['📹', '📱', '实时', 'zhíbō', 'live stream', 'broadcast'],
            '弹幕': ['💬', '📺', '评论', 'dànmù', 'bullet comments'],
            '刷礼物': ['🎁', '💰', '打赏', 'shuā lǐwù', 'send gifts'],
            '打榜': ['📊', '🚀', '投票', 'dǎbǎng', 'chart voting'],
            '控评': ['💬', '🛡️', '评论', 'kòngpíng', 'comment control'],
            '反黑': ['🛡️', '⚔️', '对抗', 'fǎnhēi', 'anti-hater'],

            '沙雕': ['🦆', '😂', '搞笑', 'shādiāo', 'funny', '逗比'],
            '逗比': ['🤡', '😂', '搞笑', 'dòubǐ', 'funny person', '沙雕'],
            '杠精': ['⚔️', '🤬', '抬杠', 'gàngjīng', 'arguer', '喜欢辩论'],
            '键盘侠': ['⌨️', '🤬', '网络暴民', 'jiànpán xiá', 'keyboard warrior'],
            '柠檬树': ['🍋', '🌳', '酸', 'níngméng shù', 'lemon tree', '嫉妒'],
            '真香': ['👃', '😋', '改口', 'zhēnxiāng', 'actually good', '打脸'],
            '鸽了': ['🕊️', '❌', '放鸽子', 'gēle', 'stood up', '取消'],
            '咕咕咕': ['🕊️', '😴', '拖延', 'gūgūgū', '鸽了', 'procrastinate'],

            'emo': ['😭', '💔', '情绪化', '伤感', 'emotional', '抑郁'],
            '丧': ['😞', '💀', '消极', 'sàng', 'depressed', '颓废'],
            '焦虑': ['😰', '💭', '担心', 'jiāolǜ', 'anxious', '紧张'],
            '社恐': ['😨', '🙈', '社交恐惧', 'shèkǒng', 'social anxiety'],
            '尬': ['😬', '😅', '尴尬', 'gà', 'awkward', '尴尬'],
            '尴尬': ['😬', '😅', '不自然', 'gāngà', 'awkward', '尬'],
            '蚌埠住了': ['🐚', '😂', '绷不住', 'bèngbù zhùle', 'can\'t hold back'],
            '破防': ['🛡️', '💥', '情绪崩溃', 'pòfáng', 'emotional breakdown'],

            '鬼畜': ['👻', '🎵', '魔性', 'guǐchù', 'weird video', 'brainwash'],
            '洗脑': ['🧠', '🔄', '重复', 'xǐnǎo', 'brainwash', '魔性'],
            '魔性': ['👿', '🔄', '上头', 'móxìng', 'addictive', '洗脑'],
            '上头': ['🤯', '🔥', '兴奋', 'shàngtóu', 'excited', '上瘾'],
            '有毒': ['☠️', '🤮', '魔性', 'yǒudú', 'toxic', 'addictive'],
            '中毒': ['☠️', '🤢', '上瘾', 'zhòngdú', 'addicted', '着迷'],

            '哎呀': ['😅', '🤦‍♀️', 'āiyā', 'oh my', '感叹'],
            '卧槽': ['😱', '🤬', 'wòcáo', 'holy shit', 'WTF'],
            '我去': ['😲', '🤯', 'wǒqù', 'damn', '惊讶'],
            '天哪': ['😱', '🙄', 'tiānnǎ', 'oh god', 'OMG'],
            '妈耶': ['😱', '🤭', 'māyé', 'oh my', 'gosh'],
            '我靠': ['😲', '🤬', 'wǒkào', 'damn', '卧槽'],
            '嘤嘤嘤': ['😭', '🥺', 'yīngyīngyīng', '哭声', 'crying sound'],
            '呜呜呜': ['😭', '😢', 'wūwūwū', '哭声', 'crying'],
            '嘿嘿嘿': ['😏', '😈', 'hēihēihēi', '坏笑', 'evil laugh'],

            }

        self.cultural_sensitive_map = {
            '河蟹': ['🦀', '和谐', 'héxiè', 'censorship', '审查'],
            '和谐': ['☮️', '🦀', 'héxié', 'harmony', '河蟹'],
            'GFW': ['🧱', '🚫', 'Great Firewall', '防火墙'],
            'VPN': ['🔒', '🌐', 'proxy', '翻墙', '代理'],
            '翻墙': ['🧗‍♂️', '🧱', 'fānqiáng', 'bypass', 'VPN'],
            '敏感词': ['⚠️', '🚫', 'mǐngǎn cí', 'sensitive word'],
            '404': ['❌', '🚫', '页面不存在', 'page not found'],
            '501': ['❌', '🚫', '服务器错误', 'server error'],
            '屏蔽': ['🚫', '❌', 'píngbì', 'block', 'ban'],
            '封号': ['🔒', '❌', 'fēnghào', 'account ban'],
            '小黑屋': ['🏠', '🔒', 'xiǎo hēiwū', 'banned room'],
            }

        self.regional_cultural_map = {
            # Northeast dialect
            '嘎嘎': ['😂', '🦆', 'gāgā', '非常', 'very'],
            '贼': ['😎', '🔥', 'zéi', '很', 'very'],
            '老铁': ['👥', '💪', 'lǎotiě', '哥们', 'bro'],
            '没毛病': ['👌', '✅', 'méi máobìng', 'no problem', '正确'],
            '整挺好': ['👍', '😎', 'zhěng tǐng hǎo', '很不错', 'pretty good'],

            # Sichuan/Southwest
            '巴适': ['😌', '👌', 'bāshì', '舒服', 'comfortable'],
            '安逸': ['😌', '☕', 'ānyì', '舒服', 'comfortable'],
            '毛线': ['🧶', '❌', 'máoxiàn', '什么', 'what the hell'],

            # Cantonese influence
            '靓仔': ['😎', '👑', 'liàngzǎi', '帅哥', 'handsome guy'],
            '靓女': ['😍', '👸', 'liàngnǚ', '美女', 'beautiful girl'],
            '搞乜': ['🤔', '❓', 'gǎomǐe', '干什么', 'what are you doing'],

            # Taiwan/Hong Kong internet culture
            '超棒': ['🔥', '👏', 'chāobàng', '很棒', 'awesome'],
            '好棒棒': ['👏', '🎉', 'hǎo bàngbàng', '很好', 'very good'],
            '歪腰': ['🤪', '😜', 'wāiyāo', '歪', 'weird'],
            '尬聊': ['😬', '💬', 'gàliáo', '尴尬聊天', 'awkward chat'],
            '小确幸': ['☕', '😌', 'xiǎo quèxìng', '小幸福', 'small happiness'],
            }
        # Latest Chinese Social Media Slang (2023-2025)
        self.latest_social_media_map = {
            # Gen Z and Young Adult Expressions
            '集美': ['👭', '💅', 'jímēi', 'sisters', '姐妹', 'bestie'],
            '绝绝子': ['🔥', '💯', '👑', 'juéjuézi', '太棒了', 'amazing'],
            '爷青回': ['😭', '⏰', '🎮', 'yéqīnghuí', '爷的青春回来了', 'nostalgia'],
            '爷青结': ['😭', '💔', '⏰', 'yéqīngjié', '爷的青春结束了', 'end of era'],
            '毁灭吧': ['💥', '😤', '🌍', 'huǐmiè ba', '算了', 'destroy it all'],
            '麻了': ['😵', '🤯', '😑', 'mále', '麻木了', 'numb'],
            '栓Q': ['🔒', '😢', 'shuānQ', 'thank you', '谢谢'],
            'YYDS': ['👑', '🔥', '💎', '永远的神', 'eternal god', '最强'],
            'DDDD': ['😂', '🤣', '💀', '笑死了', 'dying of laughter'],
            'XSWL': ['😂', '🤣', '💀', '笑死我了', 'laughing to death'],
            'AWSL': ['😍', '💖', '😵', '啊我死了', 'I\'m dead (from cuteness)'],
            'WZWZ': ['👌', '💪', '🔥', '无敌无敌', 'invincible'],
            'CPDD': ['💑', '❤️', '👫', 'couple', '组CP', 'find partner'],
            'KTV': ['🎤', '🎵', '🍻', 'karaoke', '唱歌', 'singing'],

            # Recent Viral Expressions
            '干饭人': ['🍚', '🥢', '😋', 'gànfàn rén', '吃货', 'foodie'],
            '打工人': ['👷‍♂️', '💼', '😭', 'dǎgōng rén', '上班族', 'worker'],
            '尾款人': ['💳', '💸', '😭', 'wěikuǎn rén', '付尾款的人', 'final payment person'],
            '显眼包': ['👀', '🎯', '🤡', 'xiǎnyǎn bāo', '很显眼', 'attention seeker'],
            '社死': ['💀', '😱', '🙈', 'shèsǐ', '社会性死亡', 'social death'],
            '社死现场': ['💀', '😱', '🎬', 'shèsǐ xiànchǎng', '尴尬现场', 'cringe scene'],
            'CPU': ['🧠', '🔥', '💻', 'Central Processing Unit', '脑子', 'brain'],
            'PUA': ['🤬', '💔', '🎭', 'pick up artist', '精神控制', 'manipulation'],
            '栓Q': ['🔒', '😢', 'shuānQ', 'thank you', '谢谢', '3Q', '桑Q', '伤Q'],
            '毁灭吧': ['💥', '😤', '🌍', 'huǐmiè ba', '算了', 'destroy it all', '完了', '不玩了'],
            '爷青回': ['😭', '⏰', '🎮', 'yéqīnghuí', '爷的青春回来了', 'nostalgia', '青春回来了'],
            '爷青结': ['😭', '💔', '⏰', 'yéqīngjié', '爷的青春结束了', 'end of era', '青春结束了'],
            '芭比Q了': ['💀', '🔥', '😵', 'bābǐQ le', 'barbecue', '完蛋了', 'over了', 'gg了'],
            '雀食': ['🐦', '🍚', '👍', 'què shí', '确实', 'indeed', '的确', '真的'],
            '鸡你太美': ['🐔', '🏀', '🎵', 'jī nǐ tài měi', 'meme', '梗', '只因', '因为'],
            '只因': ['🐔', '🎵', '😂', 'zhǐyīn', 'because', '原因', '就因为', '鸡你太美'],
            '孤勇者': ['🦸‍♂️', '👑', '🎵', 'gūyǒngzhě', 'lone hero', '英雄', '战士'],
            '小黑子': ['⚫', '🤬', '👿', 'xiǎo hēizi', 'hater', '黑粉', '键盘侠'],
            '显眼包': ['👀', '🎯', '🤡', 'xiǎnyǎn bāo', '很显眼', 'attention seeker', '出风头'],
            '尾款人': ['💳', '💸', '😭', 'wěikuǎn rén', '付尾款的人', 'final payment person', '买买买'],
            '干饭人': ['🍚', '🥢', '😋', 'gànfàn rén', '吃货', 'foodie', '恰饭人'],
            '打工人': ['👷‍♂️', '💼', '😭', 'dǎgōng rén', '上班族', 'worker', '社畜'],
            '电子榨菜': ['📱', '🥬', '🍜', 'diànzǐ zhàcài', '下饭视频', 'background entertainment', '配菜'],
            '人间清醒': ['🧠', '✨', '😌', 'rénjiān qīngxǐng', '很理智', 'sober minded', '清醒'],
            '人间不值得': ['🌍', '😔', '💔', 'rénjiān bù zhíde', '生活无意义', 'life not worth it', '丧'],
            '生活不易': ['😔', '💔', '🌧️', 'shēnghuó bù yì', '生活艰难', 'life is hard', '不容易'],
            '猫猫头': ['🐱', '🤔', '❓', 'māomāo tóu', '疑惑表情', 'confused cat face', '困惑'],
            '社牛': ['🐄', '👥', '🎉', 'shèniú', '社交牛逼', 'socially confident', '外向'],
            '社恐': ['😨', '🙈', '🏠', 'shèkǒng', '社交恐惧', 'socially anxious', '内向'],
            'i人': ['🏠', '😌', '📚', 'introvert', '内向', 'introverted person', 'i'],
            'e人': ['🎉', '👥', '🎤', 'extrovert', '外向', 'extroverted person', 'e'],
            '凡尔赛': ['👑', '💅', '🎭', 'fán\'ěrsài', '炫耀', 'humble bragging', '装'],
            '凡学': ['🎭', '💅', '👑', 'fán xué', '凡尔赛文学', 'Versailles literature', '装逼学'],
            '种草': ['🌱', '💚', '🛒', 'zhòng cǎo', '推荐购买', 'plant grass recommend', '安利'],
            '拔草': ['🌱', '✋', '💸', 'bá cǎo', '取消购买', 'remove from wishlist', '不买了'],
            '剁手': ['✋', '💳', '😭', 'duò shǒu', '买太多', 'overshop', '败家'],
            '薅羊毛': ['🐑', '✂️', '💰', 'háo yángmáo', '占便宜', 'get bargains', '省钱'],
            '白嫖': ['🆓', '😏', '💸', 'bái piáo', '免费获得', 'get for free', '不花钱'],
            '肝': ['😵', '⏰', '💪', 'gān', '拼命', 'grind hard', '熬夜', '努力'],
            '氪金': ['💳', '💰', '🎮', 'kē jīn', '充钱', 'pay to win', '花钱', '课金'],
            '非酋': ['💀', '😭', '🎰', 'fēi qiú', '运气差', 'bad luck', '倒霉', '黑脸'],
            '欧皇': ['👑', '🍀', '✨', 'ōu huáng', '运气好', 'good luck', '欧洲人'],
            '锦鲤': ['🐟', '🍀', '好运', 'jǐnlǐ', 'lucky', '幸运', '好彩头'],
            '咸鱼': ['🐟', '😴', '废物', 'xiányú', 'useless', '躺平', '咸鱼翻身'],
            '沙雕': ['🦆', '😂', '搞笑', 'shādiāo', 'funny', '逗比', '搞笑的'],
            '逗比': ['🤡', '😂', '搞笑', 'dòubǐ', 'funny person', '沙雕', '二货'],
            '杠精': ['⚔️', '🤬', '抬杠', 'gàngjīng', 'arguer', '喜欢辩论', '抬杠的'],
            '键盘侠': ['⌨️', '🤬', '网络暴民', 'jiànpán xiá', 'keyboard warrior', '网暴'],
            '真香': ['👃', '😋', '改口', 'zhēnxiāng', 'actually good', '打脸', '香'],
            '鸽了': ['🕊️', '❌', '放鸽子', 'gēle', 'stood up', '取消', '跳票'],
            '咕咕咕': ['🕊️', '😴', '拖延', 'gūgūgū', '鸽了', 'procrastinate', '咕咕'],
            '蚌埠住了': ['🐚', '😂', '绷不住', 'bèngbù zhùle', "can't hold back", '忍不住'],
            '有内味了': ['👃', '😋', '🎯', 'yǒu nèi wèi le', '有那个味道', 'tastes authentic', '对味'],
            '不愧是你': ['👏', '😏', '💯', 'bù kuì shì nǐ', '果然是你', 'typical of you', '还是你'],
            '针不戳': ['📍', '👌', '👍', 'zhēn bù chuō', '真不错', 'really good', '不戳'],
            '淦': ['🤬', '😤', '💢', 'gàn', '干', 'damn', 'fxxk', '草'],
            '整活': ['🎭', '😂', '🎪', 'zhěng huó', '搞笑', 'being funny', '整点活'],
            '上大分': ['📈', '🚀', '💪', 'shàng dàfēn', '提高分数', 'level up', '冲分'],
            '开摆': ['🎮', '😴', '🤷‍♂️', 'kāi bǎi', '摆烂', 'give up trying', '开始摆烂'],
            '上头': ['🤯', '🔥', '⚡', 'shàng tóu', '兴奋', 'getting excited', '激动'],
            '下头': ['😞', '📉', '💔', 'xià tóu', '失望', 'disappointed', '扫兴'],
            '有毒': ['☠️', '🤮', '魔性', 'yǒudú', 'toxic', 'addictive', '上瘾'],
            '中毒': ['☠️', '🤢', '上瘾', 'zhòngdú', 'addicted', '着迷', '沉迷'],
            '老八': ['🍔', '🤮', '💩', 'lǎobā', 'disgusting food', 'gross'],
            'emo了': ['😭', '💔', '🌧️', 'emo le', '情绪低落', 'emotional'],
            '抑郁了': ['😞', '💙', '☔', 'yìyù le', 'depressed', '难过'],
            '破大防': ['🛡️', '💥', '😭', 'pò dàfáng', '情绪崩溃', 'emotional breakdown'],
            '心态崩了': ['💥', '🤯', '😵', 'xīntài bēng le', '心理崩溃', 'mental breakdown'],
            '摆烂了': ['🗑️', '😑', '🤷‍♂️', 'bǎi làn le', '放弃了', 'gave up'],
            '躺平了': ['😴', '🛏️', '😌', 'tǎngpíng le', '不努力了', 'lie flat'],
            '卷死了': ['🌀', '😵‍💫', '📚', 'juǎn sǐ le', '太卷了', 'too competitive'],
            '绝美': ['✨', '😍', '👸', 'jué měi', '非常美', 'absolutely beautiful'],
            '神颜': ['👼', '✨', '😍', 'shén yán', '完美容貌', 'divine beauty'],
            '颜值天花板': ['📏', '👑', '✨', 'yánzhí tiānhuābǎn', '最高颜值', 'beauty ceiling'],
            '土到掉渣': ['🌍', '💩', '😅', 'tǔ dào diào zhā', '很土', 'extremely tacky'],
            '舌尖上的': ['👅', '🍽️', '📺', 'shéjiān shàng de', '美食', 'delicious food'],
            '碳水': ['🍚', '🍞', '🥖', 'tànshuǐ', '碳水化合物', 'carbohydrates'],
            '刮油': ['🫖', '🥗', '💚', 'guāyóu', '去油腻', 'cut grease'],

            '母胎solo': ['👶', '💔', '😭', 'mǔtāi solo', '单身', 'forever single'],
            '恋爱脑': ['💕', '🧠', '😵‍💫', 'liàn\'ài nǎo', '只想恋爱', 'love-obsessed'],
            'be': ['💔', '😭', '👋', 'break up', '分手', 'broken up'],
            'he': ['💕', '😍', '👫', 'happy ending', '圆满结局', 'happy ending'],
            'be美学': ['💔', '🎭', '😢', 'be měixué', '悲剧美学', 'tragic beauty'],
            '发糖': ['🍬', '💕', '😍', 'fā táng', '撒狗粮', 'sweet moments'],
            '嗑糖': ['🍬', '😋', '💕', 'kē táng', '看甜蜜', 'enjoying sweetness'],
            '发刀': ['🔪', '💔', '😭', 'fā dāo', '虐心', 'heartbreaking'],
            '上分': ['📈', '🎮', '💪', 'shàng fēn', '提高rank', 'rank up'],
            '肝游戏': ['🎮', '😵', '⏰', 'gān yóuxì', '熬夜玩', 'grinding game'],
            '土豪': ['💰', '👑', '🤑', 'tǔháo', '有钱人', 'rich person'],
            '淘宝体': ['📦', '💬', '🛒', 'táobǎo tǐ', '网购用语', 'shopping speak'],
            '学习使我快乐': ['📚', '😊', '✨', 'xuéxí shǐ wǒ kuàilè', '爱学习', 'study makes me happy'],
            '今天又是元气满满的一天': ['☀️', '💪', '😊', 'jīntiān yòu shì yuánqì mǎnmǎn de yītiān', '充满活力'],
            '冲冲冲': ['🚀', '💪', '⚡', 'chōng chōng chōng', '加油', 'go go go'],
            'ddl': ['⏰', '😱', '📝', 'deadline', '截止日期', 'due date'],
            'gg': ['💀', '😵', '结束', 'game over', '完了', 'finished'],
            '我酸了': ['🍋', '😤', '💔', 'wǒ suān le', '嫉妒了', 'I\'m jealous'],
            '柠檬了': ['🍋', '😤', '💛', 'níngméng le', '嫉妒了', 'feeling sour'],
            '哭唧唧': ['😭', '🥺', '💧', 'kū jī jī', '哭泣声', 'crying sound'],
            '无脑刷': ['🧠', '📱', '🔄', 'wúnǎo shuā', '盲目刷屏', 'mindless scrolling'],
            '算法': ['🤖', '📊', '🎯', 'suànfǎ', '推荐算法', 'algorithm'],
            '推送': ['📤', '📱', '🎯', 'tuīsòng', '推荐内容', 'push notification'],
            '流量': ['📈', '👀', '💰', 'liúliàng', '关注度', 'traffic/attention'],
            '带货': ['📦', '💰', '🛒', 'dàihuò', '直播卖货', 'live selling'],
            '直播间': ['📹', '🎤', '👥', 'zhíbòjiān', '直播房间', 'live room'],
            '弹幕': ['💬', '📺', '💭', 'dànmù', '评论弹幕', 'bullet comments'],
            '刷礼物': ['🎁', '💰', '📱', 'shuā lǐwù', '送礼物', 'send gifts'],
            '精神内耗': ['🧠', '⚡', '😵', 'jīngshén nèihào', '心理消耗', 'mental exhaustion'],
            '情绪稳定': ['😌', '⚖️', '💚', 'qíngxù wěndìng', '心态平和', 'emotionally stable'],
            '心理建设': ['🧠', '🏗️', '💪', 'xīnlǐ jiànshè', '心理准备', 'mental preparation'],
            '自我感动': ['😭', '🎭', '💔', 'zìwǒ gǎndòng', '自己感动自己', 'self-moved'],
            '破防了': ['🛡️', '💥', '😭', 'pò fáng le', '情绪崩溃', 'defenses broken'],
            '治愈': ['💚', '✨', '🌿', 'zhìyù', '心灵治疗', 'healing'],
            '治愈系': ['💚', '🌸', '😌', 'zhìyù xì', '温暖治愈', 'healing type'],

            # Time and Age References
            '后浪': ['🌊', '👶', '🆕', 'hòu làng', '年轻一代', 'younger generation'],
            '前浪': ['🌊', '👴', '📜', 'qián làng', '老一代', 'older generation'],
            '80后': ['👨‍💼', '📅', '🏢', '80 hòu', '80年代生', 'born in 80s'],
            '90后': ['👨‍💻', '📱', '🎮', '90 hòu', '90年代生', 'born in 90s'],
            '00后': ['👨‍🎓', '📱', '🎵', '00 hòu', '00年代生', 'born in 2000s'],
            '10后': ['👶', '📱', '🎮', '10 hòu', '10年代生', 'born in 2010s'],
            'Z世代': ['👨‍🎓', '📱', '🌍', 'Z shìdài', 'Z一代', 'Generation Z'],

            # Work and Career
            '996福报': ['💼', '😭', '⏰', '996 fúbào', '加班文化', '996 work culture'],
            '007': ['💼', '😱', '🌙', '全天候工作', '24/7 work'],
            '摸鱼': ['🐟', '😴', '💻', 'mōyú', '偷懒', 'slacking off'],
            '划水': ['🏊‍♂️', '😴', '💻', 'huáshuǐ', '不认真工作', 'coasting'],
            '内卷': ['🌀', '😵‍💫', '📈', 'nèijuǎn', '过度竞争', 'involution'],
            '躺平': ['😴', '🛏️', '🤷‍♂️', 'tǎngpíng', '不努力', 'lying flat'],
            '佛系': ['🧘‍♂️', '😌', '☯️', 'fóxì', '随缘', 'Buddhist mindset'],
        }

    def _apply_light_emoji(self, text: str) -> str:
        """Light emoji augmentation - conservative character replacement"""
        chars = list(text)
        replaceable_indices = [i for i, char in enumerate(chars) if char in self.char_emoji_map]

        if not replaceable_indices:
            # Try social media slang
            for slang, replacements in self.social_media_slang.items():
                if slang in text and random.random() < 0.3:
                    replacement = random.choice(replacements)
                    text = text.replace(slang, replacement, 1)
                    break

            # Try phonetic replacement
            for pinyin, emojis in self.phonetic_emoji_map.items():
                if pinyin in text.lower() and random.random() < 0.3:
                    emoji = random.choice(emojis)
                    text = text.lower().replace(pinyin, emoji, 1)
                    break
            return text

        # Replace 1-2 characters at most
        num_replacements = min(2, len(replaceable_indices), max(1, len(text) // 12))
        indices_to_replace = random.sample(replaceable_indices, num_replacements)

        for idx in indices_to_replace:
            char = chars[idx]
            replacement = random.choice(self.char_emoji_map[char])
            chars[idx] = replacement

        return ''.join(chars)

    def _apply_deep_emoji(self, text: str) -> str:
        """Deep emoji augmentation - aggressive replacement"""
        chars = list(text)

        # Higher rate character replacement
        for i, char in enumerate(chars):
            if char in self.char_emoji_map and random.random() < 0.5:
                replacement = random.choice(self.char_emoji_map[char])
                chars[i] = replacement

        text_modified = ''.join(chars)

        # Social media slang substitutions
        for slang, replacements in self.social_media_slang.items():
            if slang in text_modified and random.random() < 0.6:
                replacement = random.choice(replacements)
                text_modified = text_modified.replace(slang, replacement, 1)

        # Phonetic substitutions
        for pinyin, emojis in self.phonetic_emoji_map.items():
            if pinyin in text_modified.lower() and random.random() < 0.4:
                emoji = random.choice(emojis)
                text_modified = text_modified.lower().replace(pinyin, emoji, 1)

        # Number replacements
        for num, replacements in self.number_emoji_map.items():
            if num in text_modified and random.random() < 0.6:
                replacement = random.choice(replacements)
                text_modified = text_modified.replace(num, replacement, 1)

        # Sensitive word patterns
        for word, replacements in self.sensitive_patterns.items():
            if word in text_modified and random.random() < 0.7:
                replacement = random.choice(replacements)
                text_modified = text_modified.replace(word, replacement)

        return text_modified

    def _apply_mixed_emoji(self, text: str) -> str:
        """Mixed emoji augmentation - comprehensive approach"""
        # Start with light augmentation
        text = self._apply_light_emoji(text)

        # Add regional variations
        for word, replacements in self.regional_cultural_map.items():
            if word in text and random.random() < 0.4:
                replacement = random.choice(replacements)
                text = text.replace(word, replacement, 1)

        # Add cultural sensitive patterns
        for word, replacements in self.cultural_sensitive_map.items():
            if word in text and random.random() < 0.5:
                replacement = random.choice(replacements)
                text = text.replace(word, replacement, 1)

        # Add decorative elements around key content
        sensitive_indicators = ['批评', '抗议', '政府', '官员', '不满', '愤怒', '反对']
        for indicator in sensitive_indicators:
            if indicator in text and random.random() < 0.3:
                decorative_type = random.choice(list(self.decorative_emojis.keys()))
                decorative_emoji = random.choice(self.decorative_emojis[decorative_type])
                text = text.replace(indicator, f'{decorative_emoji}{indicator}{decorative_emoji}')

        # Add spacing and formatting variations
        if random.random() < 0.2:
            words = text.split()
            if len(words) > 2:
                insert_pos = random.randint(1, len(words) - 1)
                spacing_emoji = random.choice(['✨', '💫', '⭐', '🌟'])
                words.insert(insert_pos, spacing_emoji)
                text = ' '.join(words)

        return text

    def _apply_contextual_emoji(self, text: str) -> str:
        """Context-aware emoji application based on text content"""
        # Detect emotional context
        positive_words = ['开心', '高兴', '喜欢', '爱', '好', '棒', '赞', '666', '牛逼']
        negative_words = ['难过', '生气', '讨厌', '恨', '坏', '烂', '垃圾', '废物']

        has_positive = any(word in text for word in positive_words)
        has_negative = any(word in text for word in negative_words)

        if has_positive and random.random() < 0.4:
            celebration_emoji = random.choice(self.decorative_emojis['celebration'])
            text = text + celebration_emoji
        elif has_negative and random.random() < 0.4:
            warning_emoji = random.choice(self.decorative_emojis['warning'])
            text = warning_emoji + text

        return text

    def augment(self, text: str, intensity: str = None) -> str:
        """Apply emoji augmentation based on intensity level"""
        if intensity is None:
            intensity = self.intensity

        original_text = text.strip()
        if not original_text:
            return text

        try:
            if intensity == 'light':
                result = self._apply_light_emoji(original_text)
            elif intensity == 'deep':
                result = self._apply_deep_emoji(original_text)
            elif intensity == 'mixed':
                result = self._apply_mixed_emoji(original_text)
            else:
                result = self._apply_light_emoji(original_text)

            # Apply contextual enhancements
            result = self._apply_contextual_emoji(result)

            return result if result.strip() else original_text

        except Exception as e:
            logger.warning(f"Emoji augmentation failed: {e}")
            return original_text

    def batch_augment(self, texts: List[str], intensity: str = None) -> List[str]:
        """Apply emoji augmentation to a batch of texts"""
        return [self.augment(text, intensity) for text in texts]

    def get_mapping_stats(self) -> Dict[str, int]:
        """Return statistics about mapping dictionary sizes"""
        return {
            'char_emoji_map': len(self.char_emoji_map),
            'phonetic_emoji_map': len(self.phonetic_emoji_map),
            'social_media_slang': len(self.social_media_slang),
            'sensitive_patterns': len(self.sensitive_patterns),
            'regional_cultural_map': len(self.regional_cultural_map),
            'cultural_sensitive_map': len(self.cultural_sensitive_map),
            'total_mappings': (len(self.char_emoji_map) + len(self.phonetic_emoji_map) +
                               len(self.social_media_slang) + len(self.sensitive_patterns) +
                               len(self.regional_cultural_map) + len(self.cultural_sensitive_map))
        }
