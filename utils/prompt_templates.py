entity_prompt = """
**Objective:** Extract structured information from text chunks to build a knowledge graph.

**Your Role:** As a top-tier information extraction algorithm, your task is to identify and extract entities from provided text based on a predefined list of entity types.

**Input:**
1. **Text:** A segment of text potentially relevant to airports.
2. **Allowed Entity Types:** [Airport, Aircraft, RunwaySurfaceType, County, City, Denonym, Place, Battle, Country, Division, Language, Party, AreaCode, PostalCode, Title, Date, Class, Organisation, Person, Elevation, Length, RunwayName, ICAOCode, Order, State]
3. **Entity Type descriptions:
    *Airport*: A location where aircraft take off and land, with facilities for passengers and cargo.
    *Aircraft*: Flying vehicles designed for military operations (e.g., combat, transport) or civilian passenger and cargo transportation.
    *RunwaySurfaceType*: The material composition of a runway's surface.
    *County*: A geographic and administrative division within a state or country.
    *City*: A populated area, typically incorporating surrounding suburbs, recognized as a distinct administrative unit by a country or state/province.
    *Denonym*: A word that denotes the nationality or origin of a person or thing.
    *Place*: A physical location, including geographic areas, buildings, or specific points of interest.
    *Battle*: A fight or combat between armed forces or groups. 
    *Country*: A geopolitical entity representing a nation with its own government, occupying a particular territory.
    *Division*: A primary taxonomic rank below the Kingdom Plantae, grouping plants with shared fundamental characteristics.
    *Language*: A system of communication using symbols, sounds, or signs to convey meaning.
    *Party*: An organized group of people with shared political beliefs, aiming to influence government policy and leadership.
    *AreaCode*: A numerical code used to identify a specific geographic region for telephone calls.
    *PostalCode*: A series of letters or numbers used to identify a specific geographic area for mail delivery.
    *Title*: A word or honorific used to address or refer to a person, indicating rank, position, or respect.
    *Date*: A specific day or a reference to a particular point in time, including day, month, and year. 
    *Class*: A category or a group of similar objects, concepts, or entities. It often denotes a structured grouping that shares common attributes or behaviors.
    *Organisation*: A company or enterprise, including all its departments, employees, resources, and activities, structured to achieve specific business objectives.
    *Person*: An individual human being, typically with identifiable characteristics such as name, age, gender, and other personal attributes.
    *Elevation*: The height or vertical distance of a point or object above a specific reference level, typically above sea level.
    *Length*: The measurement of the longest dimension of an object or distance between two points.
    *RunwayName*: Refers to the designated identifier or name of a specific runway at an airport.
    *ICAOCode*: A four-letter alphanumeric code assigned by the International Civil Aviation Organization (ICAO) to uniquely identify airports, aerodromes, or airfields worldwide.
    *Order*: Refers to a rank or level in a hierarchical classification system. It is used to group related families of organisms, such as plants or animals, based on shared characteristics.
    *State*: A political entity or organization that governs a specific territory and population.
    

**Output:** A list of extracted entities, adhering to the following rules:
1. **Type Adherence:** Extract *only* entities belonging to the provided list of entity types.  Any information not assignable to one of these types *must* be ignored.
2. **Naming Convention:** Replace all spaces within extracted entity names with underscores (_).  For example, "power supply" should be extracted as "power_supply".
3. **Consistency:** Maintain consistency in entity identification. If an entity is referred to by different names or pronouns (e.g., "John_Doe", "Joe", "he"), consistently use the most complete and unambiguous identifier ("John_Doe" in this example) throughout the entire extraction process.
4. **Strict Adherence:**  Strict adherence to these rules is critical. Non-compliance will result in termination.
"""

relationship_prompt = """
**Objective:** Extract structured information from text chunks to build a knowledge graph.

**Your Role:** As a top-tier information extraction algorithm, your task is to identify relationships from the provided text based on a predefined list of relationship types and a given set of entities.

**Input:**
1. **Text:** A segment of text potentially relevant to airports.
2. **Allowed Relationship Types:** [aircraftFighter, aircraftHelicopter, runwayName, areaCode, 3rdRunwaySurfaceType, hubAirport, elevationAboveTheSeaLevelInMetres, ceremonialCounty, capital, runwaySurfaceType, headquarter, demonym, postalCode, location, owner,
regionServed, transportAircraft, order, leaderTitle, battle, cityServed, leader, city, isPartOf, icaoLocationIdentifier, elevationAboveTheSeaLevel, 2ndRunwaySurfaceType, country, division, largestCity, language, runwayLength, operatingOrganisation, leaderParty, foundedBy, class, 1stRunwaySurfaceType, foundingYear, officialLanguage]
3. **Relationship type descriptions:
    Here are the relationship types in the requested format:  
    *aircraftFighter*: A relationship linking a military organisation to a used fighter aircraft
    *aircraftHelicopter*: A relationship linking a military organisation to a used helicopter
    *runwayName*: A relationship linking an airport to its runway name.
    *areaCode*: A relationship linking a place to its area code
    *3rdRunwaySurfaceType*: A relationship linking the third runway of an airport to the runways surface material.
    *hubAirport*: A relationship linking an airline organisation to its central airport hub
    *elevationAboveTheSeaLevelInMetres*: A relationship that connects an airport to its elevation (height) above sea level measured in meters.
    *ceremonialCounty*: A connection between a specific location and the ceremonial county it belongs to, typically used for administrative or ceremonial purposes.
    *capital*: A connection between a city and the country or region it serves as the official seat of government.
    *runwaySurfaceType*: A connection between an airport and the runway surface type
    *headquarter*: A connection between an organization or company and the location of its main office or central operations.
    *demonym*: A connection between a place and the term used to describe the people or things from that place.
    *postalCode*:  A connection between a specific location and its designated postal code for mail delivery purposes.
    *location*: A general relationship between an entity and its specific geographic place.
    *owner*: A connection between an airport and the individual or organization that possesses or has legal rights to it.
    *regionServed*: A connection between an organization or entity and the geographic area it provides services to or operates within.
    *transportAircraft*: A relationship between an aircraft and its user to transport goods like cargo or people.
    *order*: A taxonomic rank that groups related entities based on shared characteristics like plants, elements or military orders.
    *leaderTitle*: A connection between a geographic location and the title of the governing body or authority that leads or oversees it.
    *battle*: A connection between an organization or entity and the military conflicts or battles it has participated in.
    *cityServed*: Describes the relationship between an airport and the specific cities it serves.
    *leader*: Defines the relationship between an individual, group or party and the role as a leader of a particular entity, such as a nation, company, or organization.
    *city*: A connection between an entity or organization and the city where it is located.
    *isPartOf*: Indicates a hierarchical or part-whole relationship, where one entity is a part of another.
    *icaoLocationIdentifier*: Relates an airport or airfield to its unique identifier assigned by the International Civil Aviation Organization (ICAO).
    *elevationAboveTheSeaLevel*: Relates an airport to its elevation above sea level. 
    *2ndRunwaySurfaceType*: A relationship linking the second runway of an airport to the runways surface material.
    *country*: Describes a relationship between a specific geographic location and the country to which it belongs.
    *division*: A high-level classification used to group entities that share common characteristics.
    *largestCity*: A connection between a geographic region and its most populous or significant city.
    *language*: A connection between a region or country and the language spoken or used by its people or associated with it.
    *runwayLength*: A relationship that connects an airport to its physical runway length.
    *operatingOrganisation*: Relates an entity to the organization responsible for operating it.
    *leaderParty*: A connection between a geographic location and the political party that leads or holds power in that area.
    *foundedBy*: A connection between an organization or entity and the individual or group responsible for its creation.
    *class*: Relates an entity to a category or classification it belongs to.
    *1stRunwaySurfaceType*: A relationship linking the first runway of an airport to the runways surface material.
    *foundingYear*: A connection between an organization or entity and the specific year it was established or created.
    *officialLanguage*: A connection between a country or region and the language that is officially recognized for government and legal purposes. 

**Output:** A list of extracted relationships, adhering to the following rules:
1. **Type Adherence:** Extract *only* relationships belonging to the provided list of relationship types.  Any information not assignable to one of these types *must* be ignored.
2. **Strict Adherence:**  Strict adherence to these rules is critical. Non-compliance will result in termination.
"""