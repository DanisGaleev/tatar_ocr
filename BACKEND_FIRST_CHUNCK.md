# План реализации бэкенда: Первая очередь (Core Security & Admin Management)

> **Статус документа:** Рабочий технический план реализации первой очереди.  
> **Цель:** Превратить текущий прототип бэкенда «Кара куян» (Tatar OCR) в защищённый сервис с ролевой моделью (RBAC), строгой изоляцией данных (Multi-tenancy), полноценной системой аутентификации (Email + Пароль, Yandex ID, VK ID), двухфакторной защитой (Email OTP, TOTP), полнофункциональной панелью администратора (включая real-time мониторинг через WebSocket) и защитой от перебора (Brute Force / Rate Limiting).

---

## 1. Архитектурная концепция первой очереди

```
                            [ КЛИЕНТЫ: Web / Mobile ]
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │        FASTAPI APPLICATION GATEWAY       │
                   │  • RequestSizeLimiter (макс. 10MB)      │
                   │  • SecurityHeadersMiddleware            │
                   │  • RateLimiterMiddleware (SlowAPI)      │
                   │  • ActivityEventBroadcaster (Telemetry) │
                   └────────────────────┬────────────────────┘
                                        │
             ┌──────────────────────────┴──────────────────────────┐
             ▼                                                     ▼
┌───────────────────────────┐                         ┌───────────────────────────┐
│       ПУБЛИЧНЫЙ СЛОЙ      │                         │     ЗАЩИЩЁННЫЙ СЛОЙ       │
│ • POST /auth/register     │                         │ (JWT Bearer: Access Token)│
│ • POST /auth/verify-email │                         ├───────────────────────────┤
│ • POST /auth/login        │                         │ RBAC: [TEACHER, ADMIN]    │
│ • POST /auth/2fa/verify   │                         │  • /classes (изолирован)  │
│ • GET  /auth/oauth/yandex │                         │  • /assignments           │
│ • GET  /auth/oauth/vk     │                         │  • /submissions           │
│ • POST /auth/refresh      │                         │  • /analytics, /reports   │
└───────────────────────────┘                         │  • /teacher/profile       │
                                                      │  • /teacher/change-passwd │
                                                      ├───────────────────────────┤
                                                      │ RBAC: [ADMIN ONLY]        │
                                                      │  • /admin/users           │
                                                      │  • /admin/users/{id}/...  │
                                                      │  • /admin/audit-logs      │
                                                      │  • WS /admin/ws/live-feed │
                                                      └───────────────────────────┘
                                        │
                                        ▼
                   ┌─────────────────────────────────────────┐
                   │             БАЗА ДАННЫХ                 │
                   │  • users, user_sessions, oauth_accounts │
                   │  • verification_codes, audit_logs       │
                   │  • classes, students, assignments, subs │
                   └─────────────────────────────────────────┘
```

---

## 2. Схема базы данных (Модели и изменения)

В первой очереди архитектура строится на SQLAlchemy 2.0 (async), что гарантирует прозрачную работу на текущем SQLite и бесшовный переход на PostgreSQL заменой строки подключения.

### 2.1 Новая таблица: `users`
Основная сущность пользователей платформы (учителя и администраторы).

```python
# app/models/user.py
class UserModel(Base):
    __tablename__ = "users"

    id = Column(String(64), primary_key=True, index=True)         # usr_... (uuid4 hex)
    email = Column(String(255), unique=True, index=True, nullable=False)
    hashed_password = Column(String(255), nullable=True)          # Argon2id / bcrypt; null для чистого OAuth
    full_name = Column(String(255), nullable=False)               # ФИО учителя/админа
    school_name = Column(String(255), nullable=True)              # Название учебного заведения
    role = Column(String(32), default="teacher", nullable=False)  # "teacher", "admin", "superadmin"
    status = Column(String(32), default="unverified", nullable=False) # "unverified", "active", "frozen", "deleted"
    
    # Флаги подтверждения и 2FA
    email_verified = Column(Boolean, default=False, nullable=False)
    two_factor_enabled = Column(Boolean, default=False, nullable=False)
    two_factor_method = Column(String(16), default="none")        # "none", "email", "totp"
    totp_secret = Column(String(64), nullable=True)               # Base32 secret для Google Authenticator
    
    # Защита от брутфорса
    failed_login_attempts = Column(Integer, default=0, nullable=False)
    locked_until = Column(DateTime(timezone=True), nullable=True)
    
    # Даты и аудит
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    updated_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))
    last_login_at = Column(DateTime(timezone=True), nullable=True)
    last_active_at = Column(DateTime(timezone=True), nullable=True)
```

### 2.2 Новая таблица: `user_sessions`
Хранение активных сессий, отслеживание устройств и ротация Refresh-токенов.

```python
# app/models/session.py
class UserSessionModel(Base):
    __tablename__ = "user_sessions"

    id = Column(String(64), primary_key=True, index=True)         # ses_...
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    refresh_token_hash = Column(String(128), unique=True, index=True, nullable=False)
    device_info = Column(String(255), nullable=True)              # "Android 14 / App 1.0.4", "Chrome on Windows"
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(512), nullable=True)
    expires_at = Column(DateTime(timezone=True), nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
    revoked_at = Column(DateTime(timezone=True), nullable=True)   # Если отозвана админом или при logout
```

### 2.3 Новая таблица: `oauth_accounts`
Привязка внешних учетных записей Yandex ID и VK ID к внутреннему пользователю.

```python
# app/models/oauth.py
class OAuthAccountModel(Base):
    __tablename__ = "oauth_accounts"

    id = Column(String(64), primary_key=True)                     # oa_...
    user_id = Column(String(64), ForeignKey("users.id", ondelete="CASCADE"), index=True, nullable=False)
    provider = Column(String(32), nullable=False)                 # "yandex", "vk"
    provider_user_id = Column(String(128), nullable=False)        # ID пользователя в Яндексе или ВК
    provider_email = Column(String(255), nullable=True)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))

    __table_args__ = (UniqueConstraint("provider", "provider_user_id", name="uq_provider_user"),)
```

### 2.4 Новая таблица: `verification_codes`
Одноразовые 6-значные коды подтверждения (регистрация, сброс пароля, 2FA через почту).

```python
# app/models/verification.py
class VerificationCodeModel(Base):
    __tablename__ = "verification_codes"

    id = Column(String(64), primary_key=True)                     # vcd_...
    email = Column(String(255), index=True, nullable=False)
    code = Column(String(16), nullable=False)                     # 6 цифр (например, "839201")
    code_type = Column(String(32), nullable=False)                # "register_verify", "login_2fa", "password_reset"
    attempts_left = Column(Integer, default=5, nullable=False)    # Защита от перебора
    expires_at = Column(DateTime(timezone=True), nullable=False)  # TTL: 10 минут
    is_used = Column(Boolean, default=False, nullable=False)
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc))
```

### 2.5 Новая таблица: `audit_logs`
Неизменяемый журнал аудита действий пользователей и администраторов.

```python
# app/models/audit.py
class AuditLogModel(Base):
    __tablename__ = "audit_logs"

    id = Column(String(64), primary_key=True)                     # aud_...
    user_id = Column(String(64), ForeignKey("users.id", ondelete="SET NULL"), index=True, nullable=True)
    user_email = Column(String(255), nullable=True)
    role = Column(String(32), nullable=True)
    action = Column(String(64), nullable=False, index=True)       # "LOGIN", "FREEZE_USER", "DELETE_STUDENT", "CLEAR_USER_DATA", etc.
    target_entity = Column(String(64), nullable=True)             # "user", "student", "class", "submission"
    target_id = Column(String(64), nullable=True)                 # ID изменяемого объекта
    ip_address = Column(String(64), nullable=True)
    user_agent = Column(String(255), nullable=True)
    details_json = Column(Text, nullable=True)                    # JSON с метаданными действия
    created_at = Column(DateTime(timezone=True), default=lambda: datetime.now(timezone.utc), index=True)
```

### 2.6 Модификация существующих моделей
1. **`ClassModel` (`school.py`)**:
   - Поле `teacher_uuid` связывается с `users.id` (ForeignKey с каскадом).
   - Индексируется для ускорения выборки классов конкретного учителя.
2. **`StudentModel` (`school.py`)**:
   - Добавляется поле `is_active = Column(Boolean, default=True)` для безопасного архивирования/удаления.
3. **`SubmissionModel` (`submission.py`)**:
   - Добавляется поле `teacher_id = Column(String(64), ForeignKey("users.id"), index=True, nullable=True)` для прямого контроля авторства проверок.
4. **`TeacherModel` (`teacher.py`)**:
   - Переводится в расширенный профиль настроек учителя (оценочные шкалы, пороги флагов) с привязкой `user_id = Column(String(64), ForeignKey("users.id"))`.

---

## 3. Детальная спецификация API первой очереди

### 3.1 Модуль Аутентификации и 2FA (`/api/v1/auth`)

| Метод | Путь | Описание | Доступ |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/auth/register` | Регистрация (email, password, ФИО, школа) -> генерация OTP кода на почту | Public |
| `POST` | `/api/v1/auth/verify-email` | Подтверждение почты кодом -> статус `active`, выдача пары JWT | Public |
| `POST` | `/api/v1/auth/resend-code` | Повторная отправка кода (с ограничением 1 раз в 60 сек) | Public |
| `POST` | `/api/v1/auth/login` | Вход по паролю -> если 2FA выключен: возврат JWT; если 2FA включен: возврат `requires_2fa=true` и `temp_token` | Public |
| `POST` | `/api/v1/auth/2fa/verify` | Подтверждение 2FA (email-код или TOTP 6 цифр) -> выдача пары JWT | Public (с temp_token) |
| `POST` | `/api/v1/auth/2fa/setup` | Генерация TOTP секрета и QR-кода (Base64) для Google Authenticator | Авторизован |
| `POST` | `/api/v1/auth/2fa/confirm-setup` | Проверка первого TOTP кода и активация 2FA + выдача резервных кодов | Авторизован |
| `POST` | `/api/v1/auth/2fa/disable` | Отключение 2FA (требует подтверждения текущим паролем) | Авторизован |
| `GET` | `/api/v1/auth/oauth/yandex/authorize` | Редирект на форму входа Яндекс ID | Public |
| `POST` | `/api/v1/auth/oauth/yandex/callback` | Обмен кода Яндекса на профиль -> логин / авторегистрация учителя | Public |
| `GET` | `/api/v1/auth/oauth/vk/authorize` | Редирект на форму входа VK ID | Public |
| `POST` | `/api/v1/auth/oauth/vk/callback` | Обмен токена VK ID на профиль -> логин / авторегистрация учителя | Public |
| `POST` | `/api/v1/auth/refresh` | Ротация Access токена по Refresh токену | Public (с Refresh) |
| `POST` | `/api/v1/auth/logout` | Отзыв текущей сессии и Refresh токена | Авторизован |
| `POST` | `/api/v1/auth/forgot-password` | Запрос 6-значного кода сброса пароля на email | Public |
| `POST` | `/api/v1/auth/reset-password` | Ввод кода сброса + установка нового пароля | Public |
| `GET` | `/api/v1/auth/me` | Данные текущего пользователя, его роль, статус и настройки | Авторизован |

---

### 3.2 Модуль Администратора (`/api/v1/admin`) — Только для `role == 'admin'`

| Метод | Путь | Описание |
| :--- | :--- | :--- |
| `GET` | `/api/v1/admin/users` | Список пользователей с пагинацией, фильтрацией по статусу (`active`, `frozen`), роли и поиском по ФИО/email |
| `GET` | `/api/v1/admin/users/{user_id}` | Детальная карточка учителя + статистика (классы, ученики, тесты, проверки, история входов) |
| `POST` | `/api/v1/admin/users/{user_id}/freeze` | **Заморозка пользователя**: статус `frozen`, мгновенный отзыв всех сессий и токенов |
| `POST` | `/api/v1/admin/users/{user_id}/unfreeze` | **Разморозка пользователя**: возврат статуса `active` |
| `DELETE` | `/api/v1/admin/users/{user_id}` | **Удаление пользователя**: деактивация/soft delete с анонимизацией персональных данных |
| `POST` | `/api/v1/admin/users/{user_id}/reset-password` | **Сброс пароля админом**: генерация нового пароля или отправка ссылки на почту |
| `PATCH` | `/api/v1/admin/users/{user_id}/profile` | **Смена ФИО**, школы или email учителя администратором |
| `POST` | `/api/v1/admin/users/{user_id}/clear-data` | **Очистка данных учителя**: удаление всех его классов, заданий и сканов (аккаунт остаётся) |
| `POST` | `/api/v1/admin/users/{user_id}/classes/{class_id}/students` | **Добавить ученика учителю**: добавление ученика в класс конкретного учителя |
| `DELETE` | `/api/v1/admin/users/{user_id}/classes/{class_id}/students/{student_id}` | **Удалить ученика**: удаление ученика из класса учителя |
| `POST` | `/api/v1/admin/users/{user_id}/classes/{class_id}/students/{student_id}/clear-data` | **Очистить данные ученика**: сброс всех оценок и истории сканов по данному ученику |
| `GET` | `/api/v1/admin/audit-logs` | Просмотр журнала аудита с фильтрами по действию, дате, пользователю |
| `GET` | `/api/v1/admin/active-sessions` | Список текущих подключений и активных сессий пользователей |
| `DELETE` | `/api/v1/admin/active-sessions/{session_id}` | Принудительное завершение конкретной сессии («Kill session») |
| `WS` | `/api/v1/admin/ws/live-feed` | **WebSocket**: трансляция действий пользователей в реальном времени (входы, сдачи тестов, ошибки) |

---

### 3.3 Модуль Учителя (`/api/v1/teacher`) и обновленный `/classes`

| Метод | Путь | Описание | Доступ |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/v1/teacher/change-password` | Смена пароля учителем (старый пароль + новый + валидация сложности + 2FA если включен) | Teacher / Admin |
| `PATCH` | `/api/v1/teacher/profile` | Обновление ФИО, школы, настроек оценивания (пороги 5/4/3, порог флага OCR) | Teacher / Admin |
| `POST` | `/api/v1/classes/{class_id}/students` | Добавление одного ученика в класс | Владелец класса |
| `PATCH` | `/api/v1/classes/{class_id}/students/{student_id}` | Редактирование ФИО / псевдонима ученика | Владелец класса |
| `DELETE` | `/api/v1/classes/{class_id}/students/{student_id}` | Удаление ученика из класса | Владелец класса |
| `POST` | `/api/v1/classes/{class_id}/students/{student_id}/clear-data` | Очистка истории сдачи тестов и результатов конкретного ученика | Владелец класса |

> **Строгая изоляция данных (Multi-tenancy):**  
> В существующих роутерах `/classes`, `/assignments`, `/submissions`, `/analytics`, `/reports` внедряется зависимость `Depends(get_current_active_user)`.  
> Любая попытка запросить или изменить класс, не принадлежащий текущему пользователю (`class.teacher_uuid != current_user.id`), отклоняется с ошибкой `403 Forbidden`.

---

## 4. Защита от атак и сетевая безопасность (Core Hardening)

### 4.1 Rate Limiting (SlowAPI / In-Memory Token Bucket)
* **/auth/login, /auth/register, /auth/verify-email, /auth/reset-password**: максимум **5 запросов в минуту** на один IP. При превышении — `429 Too Many Requests`.
* **/auth/resend-code**: жесткий интервал **не чаще 1 раза в 60 секунд**.
* **/submissions/batch-sync, /reports**: максимум **30 запросов в минуту** на пользователя.

### 4.2 Account Lockout (Защита от перебора паролей)
* При **5 неверных вводах пароля подряд** аккаунт временно блокируется на **15 минут** (`locked_until = now() + 15 min`).
* В журнал аудита отправляется событие `BRUTE_FORCE_LOCKOUT` с IP-адресом атакующего.

### 4.3 Frozen Account Guard (Мгновенная блокировка)
* При выполнении админом действия `freeze` статус пользователя меняется на `frozen`, а все записи в `user_sessions` получают `revoked_at = now()`.
* Во всех middleware и зависимостях `get_current_active_user` проверяется `user.status == 'active'`. Замороженный пользователь не может выполнить ни одного запроса даже при наличии валидного JWT (проверка происходит на уровне токена / статуса пользователя в БД).

### 4.4 Защитные заголовки и лимиты
* `RequestSizeLimiterMiddleware`: немедленно отклоняет запросы с размером тела `Content-Length > 10MB` до загрузки в оперативную память.
* Заголовки ответов:
  * `Strict-Transport-Security: max-age=31536000; includeSubDomains`
  * `X-Content-Type-Options: nosniff`
  * `X-Frame-Options: DENY`
  * `Referrer-Policy: strict-origin-when-cross-origin`

---

## 5. Механизм Real-Time мониторинга для администратора

```
 [Учитель делает действие: вход / сдача теста / ошибка]
                        │
                        ▼
           [ FastAPI Handler / Service ]
                        │
       ┌────────────────┴────────────────┐
       ▼                                 ▼
[ Запись в AuditLogModel ]    [ EventBroadcaster ]
 (сохранение в БД надолго)      (in-memory pub/sub)
                                         │
                                         ▼
                            [ WebSocket Connection ]
                            /api/v1/admin/ws/live-feed
                                         │
                                         ▼
                             [ Экран администратора ]
                           (живая лента событий online)
```

1. **Менеджер подключений (`LiveTelemetryManager`)**:
   - Ведёт пул активных WebSocket-соединений администраторов.
   - Проверяет права админа при установлении соединения (`admin_token` в query или заголовке).
2. **Формат сообщения живой ленты (JSON)**:
   ```json
   {
     "event_id": "evt_912384",
     "timestamp": "2026-11-07T18:55:00Z",
     "action": "SUBMISSION_SYNC",
     "user_id": "usr_c56a4180",
     "user_name": "Каримова Гөлнара Илдар кызы",
     "school": "Гимназия №2",
     "ip_address": "178.205.12.44",
     "details": {
       "class_name": "7-А",
       "assignment_id": "TAT-2026-Q1",
       "submissions_count": 25,
       "average_grade": 4.6
     }
   }
   ```

---

## 6. Необходимые новые библиотеки (Dependencies)

В `pyproject.toml` добавляются минимальные надежные пакеты:
* `pyjwt[crypto] >= 2.9.0` — генерация и валидация JWT Access/Refresh токенов.
* `argon2-cffi >= 23.1.0` / `passlib[bcrypt] >= 1.7.4` — безопасное хэширование паролей.
* `pyotp >= 2.9.0` — генерация TOTP секретов, валидация 6-значных кодов аутентификаторов.
* `slowapi >= 0.1.9` — rate limiting для FastAPI.
* `aiosmtplib >= 3.0.1` — асинхронная отправка писем с кодами подтверждения через SMTP.

---

## 7. Чек-лист файлов к изменению и созданию

### Новые файлы:
1. `backend/app/models/user.py` — модель `UserModel`.
2. `backend/app/models/session.py` — модель `UserSessionModel`.
3. `backend/app/models/oauth.py` — модель `OAuthAccountModel`.
4. `backend/app/models/verification.py` — модель `VerificationCodeModel`.
5. `backend/app/models/audit.py` — модель `AuditLogModel`.
6. `backend/app/core/security.py` — хэширование паролей, кодирование/декодирование JWT, генерация OTP и TOTP.
7. `backend/app/core/email.py` — сервис генерации и отправки кодов подтверждения на email (с mock-режимом для тестов).
8. `backend/app/core/telemetry.py` — WebSocket брокер живых событий админки.
9. `backend/app/schemas/user.py` — Pydantic-схемы пользователей, профилей, 2FA.
10. `backend/app/schemas/admin.py` — Pydantic-схемы запросов и ответов админки.
11. `backend/app/routers/admin.py` — роутер админ-панели и WebSocket.
12. `backend/app/routers/teacher.py` — роутер профиля учителя и смены пароля.

### Модифицируемые файлы:
1. `backend/app/core/config.py` — секретные ключи JWT, параметры OAuth (Yandex, VK), настройки SMTP.
2. `backend/app/core/database.py` — импорт новых моделей для автоматической инициализации.
3. `backend/app/routers/auth.py` — полный рефакторинг: реализация регистрации, входа, 2FA, OAuth callback.
4. `backend/app/routers/classes.py` — внедрение строгой проверки владения классом, новые методы управления учениками (удаление, очистка данных).
5. `backend/app/routers/assignments.py`, `submissions.py`, `reports.py`, `analytics.py` — внедрение зависимости авторизации `get_current_active_user` и изоляции данных.
6. `backend/app/main.py` — подключение `RateLimiterMiddleware`, `SecurityHeadersMiddleware`, новых роутеров `/admin` и `/teacher`.
7. `backend/tests/test_auth_and_admin.py` — интеграционные тесты авторизации, 2FA, заморозки, изоляции прав и админки.

---

## 8. Критерии приемки первой очереди (Definition of Done)

1. ✅ Учитель может зарегистрироваться, получить 6-значный код на почту, подтвердить его и авторизоваться.
2. ✅ Работает авторизация через Yandex ID и VK ID.
3. ✅ Учитель может подключить 2FA через TOTP (Google Authenticator) или Email-код и безопасно входить со вторым фактором.
4. ✅ Учитель может сменить пароль в личном кабинете.
5. ✅ Учитель может полноценно управлять своими учениками (добавлять, менять ФИО, удалять, сбрасывать оценки/историю).
6. ✅ Строгая изоляция: учитель A не может просмотреть или изменить классы и работы учителя B.
7. ✅ Администратор видит список всех пользователей, может **заморозить** пользователя (с мгновенным прекращением его доступа) и **разморозить** обратно.
8. ✅ Администратор может **удалить** пользователя, **сбросить ему пароль**, **сменить ФИО** или **очистить все его данные**.
9. ✅ Администратор может управлять учениками любого учителя (добавить, удалить, очистить данные).
10. ✅ Администратор подключается к WebSocket `/admin/ws/live-feed` и видит события действий пользователей в реальном времени.
11. ✅ Срабатывает Rate Limiting при попытке подбора пароля или спама кодами.
12. ✅ Все существующие 139 тестов бэкенда продолжают успешно проходить + добавлены новые тесты для ядра безопасности.
