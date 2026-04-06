import Foundation
import CoreGraphics
import GRDB

// MARK: - Data Types

nonisolated
struct FaceRecord: Codable, FetchableRecord, PersistableRecord, Identifiable, Sendable {
    var id: Int64?
    let assetId: String
    let boundingBox: Data        // JSON-encoded CGRect
    let embedding: Data          // 512 × Float32 = 2048 bytes
    let embeddingVersion: Int
    var personId: Int?

    static let databaseTableName = "faceRecord"

    mutating func didInsert(_ inserted: InsertionSuccess) {
        id = inserted.rowID
    }
}

nonisolated
struct PersonCluster: Identifiable, Sendable, Equatable {
    let id: Int
    var centroid: [Float]
    var count: Int
    let sampleAssetId: String
    let sampleBoundingBox: CGRect
}

// MARK: - Database Manager

nonisolated
final class FaceDatabase: Sendable {
    private let dbQueue: DatabaseQueue

    init() throws {
        let url = try FileManager.default
            .url(for: .applicationSupportDirectory,
                 in: .userDomainMask,
                 appropriateFor: nil,
                 create: true)
            .appendingPathComponent("faces.sqlite")

        dbQueue = try DatabaseQueue(path: url.path)
        try migrator.migrate(dbQueue)
    }

    private var migrator: DatabaseMigrator {
        var migrator = DatabaseMigrator()
        migrator.registerMigration("v1") { db in
            try db.create(table: "faceRecord") { t in
                t.autoIncrementedPrimaryKey("id")
                t.column("assetId", .text).notNull()
                t.column("boundingBox", .blob).notNull()
                t.column("embedding", .blob).notNull()
                t.column("personId", .integer)
            }
        }
        migrator.registerMigration("v2") { db in
            try db.create(table: "personName") { t in
                t.column("personId", .integer).primaryKey()
                t.column("name", .text).notNull()
            }
        }
        migrator.registerMigration("v3") { db in
            try db.alter(table: "faceRecord") { t in
                t.add(column: "embeddingVersion", .integer).notNull().defaults(to: 1)
            }
        }
        return migrator
    }

    // MARK: - CRUD

    func insert(_ record: inout FaceRecord) throws {
        try dbQueue.write { db in
            try record.insert(db)
        }
    }

    func updatePersonId(_ id: Int64, personId: Int) throws {
        try dbQueue.write { db in
            try db.execute(
                sql: "UPDATE faceRecord SET personId = ? WHERE id = ?",
                arguments: [personId, id]
            )
        }
    }

    /// Batch-update personIds in a single transaction (much faster than individual writes).
    func batchUpdatePersonIds(_ updates: [(faceId: Int64, personId: Int)]) throws {
        try dbQueue.write { db in
            for (faceId, personId) in updates {
                try db.execute(
                    sql: "UPDATE faceRecord SET personId = ? WHERE id = ?",
                    arguments: [personId, faceId]
                )
            }
        }
    }

    func allFaces(embeddingVersion: Int? = nil) throws -> [FaceRecord] {
        try dbQueue.read { db in
            var request = FaceRecord.all()
            if let embeddingVersion {
                request = request.filter(Column("embeddingVersion") == embeddingVersion)
            }
            return try request.fetchAll(db)
        }
    }

    func faces(forPerson personId: Int) throws -> [FaceRecord] {
        try dbQueue.read { db in
            try FaceRecord
                .filter(Column("personId") == personId)
                .fetchAll(db)
        }
    }

    func faces(forAsset assetId: String) throws -> [FaceRecord] {
        try dbQueue.read { db in
            try FaceRecord
                .filter(Column("assetId") == assetId)
                .fetchAll(db)
        }
    }

    func processedAssetIds(embeddingVersion: Int? = nil) throws -> Set<String> {
        try dbQueue.read { db in
            let ids: [String]
            if let embeddingVersion {
                ids = try String.fetchAll(
                    db,
                    sql: "SELECT DISTINCT assetId FROM faceRecord WHERE embeddingVersion = ?",
                    arguments: [embeddingVersion]
                )
            } else {
                ids = try String.fetchAll(db,
                    sql: "SELECT DISTINCT assetId FROM faceRecord")
            }
            return Set(ids)
        }
    }

    @discardableResult
    func deleteAll() throws -> Int {
        try dbQueue.write { db in
            let deleted = try FaceRecord.deleteAll(db)
            try db.execute(sql: "DELETE FROM personName")
            return deleted
        }
    }

    func faceCount(embeddingVersion: Int? = nil) throws -> Int {
        try dbQueue.read { db in
            if let embeddingVersion {
                return try FaceRecord
                    .filter(Column("embeddingVersion") == embeddingVersion)
                    .fetchCount(db)
            }
            return try FaceRecord.fetchCount(db)
        }
    }

    func availableEmbeddingVersions() throws -> [Int] {
        try dbQueue.read { db in
            try Int.fetchAll(
                db,
                sql: "SELECT DISTINCT embeddingVersion FROM faceRecord ORDER BY embeddingVersion DESC"
            )
        }
    }

    func mergePersonIds(from oldId: Int, into newId: Int) throws {
        try dbQueue.write { db in
            try db.execute(
                sql: "UPDATE faceRecord SET personId = ? WHERE personId = ?",
                arguments: [newId, oldId])
        }
    }

    func firstFace(forPerson personId: Int) throws -> FaceRecord? {
        try dbQueue.read { db in
            try FaceRecord
                .filter(Column("personId") == personId)
                .fetchOne(db)
        }
    }

    /// Returns the face with the largest bounding box area for a given person.
    /// Larger bounding boxes correspond to closer/higher-resolution face crops.
    func bestFace(forPerson personId: Int) throws -> FaceRecord? {
        let faces = try self.faces(forPerson: personId)
        return faces.max(by: { a, b in
            let aRect = a.boundingRect
            let bRect = b.boundingRect
            return (aRect.width * aRect.height) < (bRect.width * bRect.height)
        })
    }

    // MARK: - Person Names

    func setName(forPerson personId: Int, name: String) throws {
        try dbQueue.write { db in
            try db.execute(
                sql: "INSERT OR REPLACE INTO personName (personId, name) VALUES (?, ?)",
                arguments: [personId, name]
            )
        }
    }

    func name(forPerson personId: Int) throws -> String? {
        try dbQueue.read { db in
            try String.fetchOne(db,
                sql: "SELECT name FROM personName WHERE personId = ?",
                arguments: [personId])
        }
    }

    func allNames() throws -> [Int: String] {
        try dbQueue.read { db in
            var result: [Int: String] = [:]
            let rows = try Row.fetchAll(db, sql: "SELECT personId, name FROM personName")
            for row in rows {
                let personId: Int = row["personId"]
                let name: String = row["name"]
                result[personId] = name
            }
            return result
        }
    }
}

// MARK: - Embedding Helpers

extension FaceRecord {
    var embeddingVector: [Float] {
        embedding.withUnsafeBytes { buffer in
            Array(buffer.bindMemory(to: Float.self))
        }
    }

    var boundingRect: CGRect {
        (try? JSONDecoder().decode(CGRect.self, from: boundingBox)) ?? .zero
    }

    static func encodeBoundingBox(_ rect: CGRect) -> Data {
        (try? JSONEncoder().encode(rect)) ?? Data()
    }

    static func encodeEmbedding(_ vector: [Float]) -> Data {
        vector.withUnsafeBufferPointer { buffer in
            Data(buffer: buffer)
        }
    }
}
