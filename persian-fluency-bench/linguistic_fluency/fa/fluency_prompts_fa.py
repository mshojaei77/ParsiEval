from __future__ import annotations


DOMAINS = [
    "health",
    "education",
    "shopping",
    "travel",
    "banking",
    "food",
    "technology",
    "career",
    "transportation",
    "entertainment",
]


def get_fa_fluency_cases() -> list[dict[str, str | list[str]]]:
    """Modern native Persian fluency cases across 10 everyday domains.

    Every domain appears in both registers:

    - ``casual``: chats, everyday talking, short replies -> expect natural
      colloquial (mahavireh) Persian.
    - ``formal``: writing books, articles, letters -> expect eloquent
      standard written Persian.
    """
    return [
        # ---------- Casual: health ----------
        {
            "id": "fa_fluency_001",
            "domain": "health",
            "tags": ["casual", "general_fluency"],
            "user": "وای دیشب تا صبح سرفه کردم، خوابم نبرد اصلاً. همین‌جوری خودم درمان کنم یا برم دکتر؟",
        },
        {
            "id": "fa_fluency_002",
            "domain": "health",
            "tags": ["casual", "general_fluency"],
            "user": "راستش از دندون‌پزشکی می‌ترسم، ولی دندون عقلم درد می‌کنه. دکتوری می‌شناسی دستش خیلی نرم باشه؟",
        },
        # ---------- Casual: education ----------
        {
            "id": "fa_fluency_003",
            "domain": "education",
            "tags": ["casual", "general_fluency"],
            "user": "پسرخاله‌ام امسال کنکور داد، رتبه‌ش به اون رشته‌ای که می‌خواست نرسید. می‌خوام یه پیام دلداری بدیم، چی بنویسم که حالش رو بگیره؟",
        },
        {
            "id": "fa_fluency_004",
            "domain": "education",
            "tags": ["casual", "general_fluency"],
            "user": "جزوه‌های فیزیک پیش‌دانشمندی رو داری؟ فردا امتحان داریم والا من هیچی بلد نیستم.",
        },
        # ---------- Casual: shopping ----------
        {
            "id": "fa_fluency_005",
            "domain": "shopping",
            "tags": ["casual", "general_fluency"],
            "user": "آقا این کاپشن آخرش چند می‌شه؟ ول کن یه تخفیفی بده که هر دو طرف راضی باشیم!",
        },
        {
            "id": "fa_fluency_006",
            "domain": "shopping",
            "tags": ["casual", "general_fluency"],
            "user": "این کتونی که آنلاین سفارش دادم یه سایز کوچیک‌تر از پام اومده. پس بدم بهتره یا مرجوع کنم؟",
        },
        # ---------- Casual: travel ----------
        {
            "id": "fa_fluency_007",
            "domain": "travel",
            "tags": ["casual", "context_retention"],
            "user": "پیام ۱: می‌خوایم آخر هفته بریم سفر. پیام ۲: بودجه‌مون کمه. پیام ۳: دو تا بچه کوچیک هم داریم. پیام ۴: ماشین هم نداریم. حالا با توجه به همه‌ی حرفام، بگو کجا و چطوری بهتره بریم؟",
        },
        {
            "id": "fa_fluency_008",
            "domain": "travel",
            "tags": ["casual", "general_fluency"],
            "user": "ببخشید آقا، واسه رفتن به ترمینال جنوب از اینجا کدوم طرف باید بریم؟ اتوبوس هم این نزدیکیا سوار می‌کنه؟",
        },
        # ---------- Casual: banking ----------
        {
            "id": "fa_fluency_009",
            "domain": "banking",
            "tags": ["casual", "general_fluency"],
            "user": "طلا باز قیمتش رفت بالا ها! به نظرت الان بخرم یا صبر کنم پایین بیاد؟",
        },
        {
            "id": "fa_fluency_010",
            "domain": "banking",
            "tags": ["casual", "general_fluency"],
            "user": "کارت به کارت کردم براش، نیم ساعت گذشته هنوز نرسیده. نگران نباشه یا برم پیگیری کنم؟",
        },
        # ---------- Casual: food ----------
        {
            "id": "fa_fluency_011",
            "domain": "food",
            "tags": ["casual", "general_fluency"],
            "user": "مهمون داریم امشب، بین قرمه‌سبزی و گوجه‌پلو گیر کردم. نظرت کدوم به‌صرفه‌تر و به‌مزه‌تره؟",
        },
        {
            "id": "fa_fluency_012",
            "domain": "food",
            "tags": ["casual", "general_fluency"],
            "user": "وای برنج آباد شد! الان دیگه چیکارش کنم؟",
        },
        # ---------- Casual: technology ----------
        {
            "id": "fa_fluency_013",
            "domain": "technology",
            "tags": ["casual", "general_fluency"],
            "user": "وای‌فای خونه‌مون هی قطع و وصل می‌شه، مودم رو هم خاموش و روشن کردم درست نشد. تو تا حالا این مشکل رو داشتی؟",
        },
        {
            "id": "fa_fluency_014",
            "domain": "technology",
            "tags": ["casual", "general_fluency"],
            "user": "حافظه‌ی گوشیم پر شده هی می‌گه فضای کافی ندارید. به نظرت اول کدوم چیزا رو پاک کنم؟",
        },
        # ---------- Casual: career ----------
        {
            "id": "fa_fluency_015",
            "domain": "career",
            "tags": ["casual", "general_fluency"],
            "user": "خوب بودی این آخر هفته؟ تعطیلات چی‌کار کردی؟",
        },
        {
            "id": "fa_fluency_016",
            "domain": "career",
            "tags": ["casual", "general_fluency"],
            "user": "باز مدیرمون گفت امروز تا ساعت ۹ اضافه‌کاری مونی. اعصابم خورد شده داداش.",
        },
        # ---------- Casual: transportation ----------
        {
            "id": "fa_fluency_017",
            "domain": "transportation",
            "tags": ["casual", "general_fluency"],
            "user": "دوباره همین ترافیک مسیر کن؟! سه‌ربع ساعته تو خیابون موندم.",
        },
        {
            "id": "fa_fluency_018",
            "domain": "transportation",
            "tags": ["casual", "general_fluency"],
            "user": "امروز صبح ماشینم با این سرما روشن نشد، کلی اذیت شدم. به نظرت باتریه یا استارت؟",
        },
        # ---------- Casual: entertainment ----------
        {
            "id": "fa_fluency_019",
            "domain": "entertainment",
            "tags": ["casual", "general_fluency"],
            "user": "دیروقت بازی رو دیدی؟ آخرش باورم نمی‌شد!",
        },
        {
            "id": "fa_fluency_020",
            "domain": "entertainment",
            "tags": ["casual", "naturalness"],
            "user": "اون سریال جدیدو دیدی؟ من دو قسمتشو رفتم ولی هنوز مطمئن نیستم ادامه بدم یا نه.",
        },
        # ---------- Formal: health ----------
        {
            "id": "fa_fluency_021",
            "domain": "health",
            "tags": ["formal", "writing_article"],
            "user": "پاراگراف آغازین مقاله‌ای دربارهٔ نقش ورزش و پیشگیری در سلامت عمومی جامعه را برای یک مجلهٔ بهداشتی بنویسید.",
        },
        # ---------- Formal: education ----------
        {
            "id": "fa_fluency_022",
            "domain": "education",
            "tags": ["formal", "writing_email"],
            "user": "ایمیلی رسمی به معاونت دانشجویی دانشگاه بنویسید و درخواست بورسیهٔ تحصیلی را با ذکر شرایط خود مطرح کنید.",
        },
        # ---------- Formal: shopping ----------
        {
            "id": "fa_fluency_023",
            "domain": "shopping",
            "tags": ["formal", "writing_letter"],
            "user": "نامهٔ شکایت رسمی به یک فروشگاه اینترنتی بنویسید؛ کالای معیوب تحویل شده و از پاسخگویی پشتیبانی نتیجه‌ای نگرفته‌اید.",
        },
        # ---------- Formal: travel ----------
        {
            "id": "fa_fluency_024",
            "domain": "travel",
            "tags": ["formal", "writing_article"],
            "user": "مقالهٔ کوتاهی برای یک نشریهٔ گردشگری دربارهٔ جذابیت‌های تاریخی اصفهان بنویسید؛ نثری رسا و جذاب.",
        },
        # ---------- Formal: banking ----------
        {
            "id": "fa_fluency_025",
            "domain": "banking",
            "tags": ["formal", "writing_letter"],
            "user": "نامه‌ای رسمی به شعبهٔ بانک بنویسید و درخواست دریافت صورت‌حساب کامل و توضیح شرایط بازپرداخت وام را مطرح کنید.",
        },
        # ---------- Formal: food ----------
        {
            "id": "fa_fluency_026",
            "domain": "food",
            "tags": ["formal", "writing_book"],
            "user": "پاراگرافی کتاب‌گونه دربارهٔ جایگاه سفرهٔ ایرانی و مهمان‌نوازی در فرهنگ ما بنویسید؛ لحن ادبی اما روان.",
        },
        # ---------- Formal: technology ----------
        {
            "id": "fa_fluency_027",
            "domain": "technology",
            "tags": ["formal", "writing_article"],
            "user": "بخشی از مقالهٔ تحلیلی دربارهٔ تأثیر هوش مصنوعی بر زندگی روزمره را برای یک روزنامهٔ صبح بنویسید؛ زبان معیار و استدلالی باشد.",
        },
        # ---------- Formal: career ----------
        {
            "id": "fa_fluency_028",
            "domain": "career",
            "tags": ["formal", "instruction_following"],
            "user": "دقیقاً در دو جملهٔ رسمی، پذیرش پیشنهاد شغل را در قالب نامهٔ کوتاه اعلام کنید.",
        },
        # ---------- Formal: transportation ----------
        {
            "id": "fa_fluency_029",
            "domain": "transportation",
            "tags": ["formal", "writing_letter"],
            "user": "نامه‌ای رسمی به مرکز راهنمایی و رانندگی بنویسید و نسبت به جریمه‌ای که برای خودروی شما ثبت شده اعتراض کنید.",
        },
        # ---------- Formal: entertainment ----------
        {
            "id": "fa_fluency_030",
            "domain": "entertainment",
            "tags": ["formal", "writing_article"],
            "user": "پاراگراف کوتاهی از مقاله‌ای دربارهٔ نقش ورزش در سلامت روان و انسجام اجتماعی بنویسید؛ نثری والا اما شیوا.",
        },
    ]
